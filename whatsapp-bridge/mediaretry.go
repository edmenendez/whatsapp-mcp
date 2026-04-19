// Media retry: ask the sender's phone to re-upload media whose CDN URL has
// expired. Used to recover images/videos that the bridge has metadata + keys
// for but whose CDN blob has been purged (history-sync media past retention).
//
// Protocol: Client.SendMediaRetryReceipt → sender's phone re-encrypts/re-uploads
// → events.MediaRetry arrives with a fresh DirectPath → DownloadMediaWithPath
// fetches the new blob → file lands on disk (triggering the watcher).

package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
)

// RetryMediaRequest is the JSON payload for POST /api/retry_media.
type RetryMediaRequest struct {
	MessageID string `json:"message_id"`
	ChatJID   string `json:"chat_jid"`
}

// RetryMediaResponse mirrors DownloadMediaResponse — consumers can treat the
// two endpoints interchangeably on the Python side.
type RetryMediaResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
}

// messageRetryInfo is everything we need from messages.db to both send the
// retry receipt and later re-download when the response arrives.
type messageRetryInfo struct {
	messageID     string
	chatJID       string
	sender        string
	isFromMe      bool
	timestamp     time.Time
	mediaType     string
	mediaKey      []byte
	fileSHA256    []byte
	fileEncSHA256 []byte
	fileLength    uint64
}

// loadRetryInfo reads the DB row required to drive a retry for a single message.
func loadRetryInfo(store *MessageStore, messageID, chatJID string) (*messageRetryInfo, error) {
	info := &messageRetryInfo{messageID: messageID, chatJID: chatJID}
	err := store.db.QueryRow(
		`SELECT sender, is_from_me, timestamp,
		        media_type, media_key, file_sha256, file_enc_sha256, file_length
		 FROM messages
		 WHERE id = ? AND chat_jid = ?`,
		messageID, chatJID,
	).Scan(
		&info.sender, &info.isFromMe, &info.timestamp,
		&info.mediaType, &info.mediaKey, &info.fileSHA256, &info.fileEncSHA256, &info.fileLength,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, fmt.Errorf("no message %s in chat %s", messageID, chatJID)
	}
	if err != nil {
		return nil, fmt.Errorf("query message: %w", err)
	}
	if info.mediaType == "" {
		return nil, fmt.Errorf("message is not media")
	}
	if len(info.mediaKey) == 0 {
		return nil, fmt.Errorf("message missing media_key — cannot retry")
	}
	return info, nil
}

// buildMessageInfo reconstructs enough of types.MessageInfo for
// SendMediaRetryReceipt. That function only reads ID, Chat, Sender,
// IsFromMe, and IsGroup.
func (m *messageRetryInfo) buildMessageInfo() (*types.MessageInfo, error) {
	chatJID, err := types.ParseJID(m.chatJID)
	if err != nil {
		return nil, fmt.Errorf("parse chat jid: %w", err)
	}
	isGroup := chatJID.Server == types.GroupServer

	var senderJID types.JID
	if m.sender != "" {
		// messages.sender stores the user part only, not a full JID.
		senderJID = types.JID{User: m.sender, Server: types.DefaultUserServer}
	}

	return &types.MessageInfo{
		MessageSource: types.MessageSource{
			Chat:     chatJID,
			Sender:   senderJID,
			IsFromMe: m.isFromMe,
			IsGroup:  isGroup,
		},
		ID:        m.messageID,
		Timestamp: m.timestamp,
	}, nil
}

// sendMediaRetryRequest looks up a message, reconstructs its MessageInfo, and
// asks the sender's phone to re-upload the media. The actual re-download
// happens later when the *events.MediaRetry event arrives.
func sendMediaRetryRequest(ctx context.Context, client *whatsmeow.Client, store *MessageStore, messageID, chatJID string) error {
	info, err := loadRetryInfo(store, messageID, chatJID)
	if err != nil {
		return err
	}
	msgInfo, err := info.buildMessageInfo()
	if err != nil {
		return err
	}
	return client.SendMediaRetryReceipt(ctx, msgInfo, info.mediaKey)
}

// handleMediaRetryEvent is called when the sender responds to a retry receipt.
// On success it decrypts the new DirectPath, re-downloads the media, and
// writes it to disk (which fires FSEvents for the watcher).
func handleMediaRetryEvent(ctx context.Context, client *whatsmeow.Client, store *MessageStore, evt *events.MediaRetry, logger interface{ Warnf(string, ...any); Infof(string, ...any) }) {
	// Messages are stored under phone JIDs, but retry notifications can
	// arrive addressed to the LID. Try the event's ChatID first; if that
	// misses and the ChatID is a LID, resolve LID→phone via whatsmeow's
	// LID store and retry the lookup.
	info, err := loadRetryInfo(store, evt.MessageID, evt.ChatID.String())
	if err != nil && evt.ChatID.Server == types.HiddenUserServer {
		if pn, lidErr := client.Store.LIDs.GetPNForLID(ctx, evt.ChatID); lidErr == nil && !pn.IsEmpty() {
			info, err = loadRetryInfo(store, evt.MessageID, pn.ToNonAD().String())
		}
	}
	if err != nil {
		logger.Warnf("media retry for %s (chat=%s): %v", evt.MessageID, evt.ChatID, err)
		return
	}

	notif, err := whatsmeow.DecryptMediaRetryNotification(evt, info.mediaKey)
	if err != nil {
		if errors.Is(err, whatsmeow.ErrMediaNotAvailableOnPhone) {
			logger.Warnf("media retry %s: sender phone no longer has the media (code 2)", evt.MessageID)
			return
		}
		logger.Warnf("media retry %s: decrypt failed: %v", evt.MessageID, err)
		return
	}

	if notif.GetResult().String() != "SUCCESS" {
		logger.Warnf("media retry %s: non-success result %s", evt.MessageID, notif.GetResult().String())
		return
	}

	directPath := notif.GetDirectPath()
	if directPath == "" {
		logger.Warnf("media retry %s: success but no DirectPath in response", evt.MessageID)
		return
	}

	waMediaType, err := mediaTypeFromString(info.mediaType)
	if err != nil {
		logger.Warnf("media retry %s: %v", evt.MessageID, err)
		return
	}

	logger.Infof("media retry %s: re-downloading from new path", evt.MessageID)
	data, err := client.DownloadMediaWithPath(
		ctx,
		directPath,
		info.fileEncSHA256,
		info.fileSHA256,
		info.mediaKey,
		int(info.fileLength),
		waMediaType,
		"", // mmsType — whatsmeow derives from waMediaType
	)
	if err != nil {
		logger.Warnf("media retry %s: download failed: %v", evt.MessageID, err)
		return
	}

	if err := writeRetriedMediaFile(info, data); err != nil {
		logger.Warnf("media retry %s: write file: %v", evt.MessageID, err)
		return
	}
	logger.Infof("media retry %s: saved %d bytes (watcher will describe)", evt.MessageID, len(data))
}

// writeRetriedMediaFile saves retried media using the same naming convention
// as the auto-download path (image_YYYYMMDD_HHMMSS_MSGID.ext).
func writeRetriedMediaFile(info *messageRetryInfo, data []byte) error {
	ext := extForMediaType(info.mediaType)
	tsStr := info.timestamp.Format("20060102_150405")
	filename := fmt.Sprintf("%s_%s_%s%s", info.mediaType, tsStr, info.messageID, ext)
	chatDir := fmt.Sprintf("store/%s", strings.ReplaceAll(info.chatJID, ":", "_"))
	if err := os.MkdirAll(chatDir, 0755); err != nil {
		return fmt.Errorf("create chat dir: %w", err)
	}
	return os.WriteFile(fmt.Sprintf("%s/%s", chatDir, filename), data, 0644)
}

func extForMediaType(mediaType string) string {
	switch mediaType {
	case "image":
		return ".jpg"
	case "video":
		return ".mp4"
	case "audio":
		return ".ogg"
	default:
		return ""
	}
}

func mediaTypeFromString(s string) (whatsmeow.MediaType, error) {
	switch s {
	case "image":
		return whatsmeow.MediaImage, nil
	case "video":
		return whatsmeow.MediaVideo, nil
	case "audio":
		return whatsmeow.MediaAudio, nil
	case "document":
		return whatsmeow.MediaDocument, nil
	default:
		return "", fmt.Errorf("unsupported media type: %s", s)
	}
}

// registerRetryMediaEndpoint wires POST /api/retry_media onto the existing
// REST server. Kept separate so this feature is isolated from the other
// endpoints in main.go.
func registerRetryMediaEndpoint(client *whatsmeow.Client, store *MessageStore) {
	http.HandleFunc("/api/retry_media", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req RetryMediaRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeRetryResponse(w, http.StatusBadRequest, false, "invalid request: "+err.Error())
			return
		}
		if req.MessageID == "" || req.ChatJID == "" {
			writeRetryResponse(w, http.StatusBadRequest, false, "message_id and chat_jid are required")
			return
		}
		if !client.IsConnected() {
			writeRetryResponse(w, http.StatusServiceUnavailable, false, "WhatsApp client not connected")
			return
		}
		if err := sendMediaRetryRequest(r.Context(), client, store, req.MessageID, req.ChatJID); err != nil {
			writeRetryResponse(w, http.StatusInternalServerError, false, err.Error())
			return
		}
		writeRetryResponse(w, http.StatusOK, true,
			"Retry receipt sent — media will download async when sender responds (event arrives as MediaRetry)")
	})
}

func writeRetryResponse(w http.ResponseWriter, status int, success bool, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(RetryMediaResponse{Success: success, Message: message})
}
