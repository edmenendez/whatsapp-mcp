CREATE TABLE IF NOT EXISTS transcriptions (
    message_id TEXT NOT NULL,
    chat_jid TEXT NOT NULL,
    transcription TEXT NOT NULL,
    language TEXT,
    model TEXT NOT NULL,
    duration_sec REAL,
    audio_sha256 TEXT,
    transcribed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (message_id, chat_jid)
);

CREATE INDEX IF NOT EXISTS idx_transcriptions_chat
    ON transcriptions(chat_jid);

CREATE VIRTUAL TABLE IF NOT EXISTS transcriptions_fts USING fts5(
    transcription,
    message_id UNINDEXED,
    chat_jid UNINDEXED,
    content=transcriptions,
    content_rowid=rowid,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS transcriptions_ai AFTER INSERT ON transcriptions BEGIN
    INSERT INTO transcriptions_fts(rowid, transcription, message_id, chat_jid)
    VALUES (new.rowid, new.transcription, new.message_id, new.chat_jid);
END;

CREATE TRIGGER IF NOT EXISTS transcriptions_ad AFTER DELETE ON transcriptions BEGIN
    INSERT INTO transcriptions_fts(transcriptions_fts, rowid, transcription, message_id, chat_jid)
    VALUES ('delete', old.rowid, old.transcription, old.message_id, old.chat_jid);
END;

CREATE TRIGGER IF NOT EXISTS transcriptions_au AFTER UPDATE ON transcriptions BEGIN
    INSERT INTO transcriptions_fts(transcriptions_fts, rowid, transcription, message_id, chat_jid)
    VALUES ('delete', old.rowid, old.transcription, old.message_id, old.chat_jid);
    INSERT INTO transcriptions_fts(rowid, transcription, message_id, chat_jid)
    VALUES (new.rowid, new.transcription, new.message_id, new.chat_jid);
END;
