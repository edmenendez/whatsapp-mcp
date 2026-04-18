CREATE TABLE IF NOT EXISTS image_descriptions (
    message_id TEXT NOT NULL,
    chat_jid TEXT NOT NULL,
    description TEXT NOT NULL,
    ocr_text TEXT,
    model TEXT NOT NULL,
    image_sha256 TEXT,
    width INTEGER,
    height INTEGER,
    described_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (message_id, chat_jid)
);

CREATE INDEX IF NOT EXISTS idx_image_descriptions_chat
    ON image_descriptions(chat_jid);

CREATE VIRTUAL TABLE IF NOT EXISTS image_descriptions_fts USING fts5(
    description,
    ocr_text,
    message_id UNINDEXED,
    chat_jid UNINDEXED,
    content=image_descriptions,
    content_rowid=rowid,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS image_descriptions_ai AFTER INSERT ON image_descriptions BEGIN
    INSERT INTO image_descriptions_fts(rowid, description, ocr_text, message_id, chat_jid)
    VALUES (new.rowid, new.description, new.ocr_text, new.message_id, new.chat_jid);
END;

CREATE TRIGGER IF NOT EXISTS image_descriptions_ad AFTER DELETE ON image_descriptions BEGIN
    INSERT INTO image_descriptions_fts(image_descriptions_fts, rowid, description, ocr_text, message_id, chat_jid)
    VALUES ('delete', old.rowid, old.description, old.ocr_text, old.message_id, old.chat_jid);
END;

CREATE TRIGGER IF NOT EXISTS image_descriptions_au AFTER UPDATE ON image_descriptions BEGIN
    INSERT INTO image_descriptions_fts(image_descriptions_fts, rowid, description, ocr_text, message_id, chat_jid)
    VALUES ('delete', old.rowid, old.description, old.ocr_text, old.message_id, old.chat_jid);
    INSERT INTO image_descriptions_fts(rowid, description, ocr_text, message_id, chat_jid)
    VALUES (new.rowid, new.description, new.ocr_text, new.message_id, new.chat_jid);
END;
