CREATE TABLE IF NOT EXISTS runs (
  id            TEXT PRIMARY KEY,   -- "run_2026-09-18T07:00"
  target_date   TEXT NOT NULL,
  started_at    TEXT NOT NULL,
  finished_at   TEXT,
  status        TEXT NOT NULL,      -- running|ok|failed|budget_exceeded|pending_review
  turns         INTEGER,
  images_sent   INTEGER,
  input_tokens  INTEGER,
  output_tokens INTEGER,
  cost_usd      REAL,
  provider      TEXT,
  model         TEXT,
  prompt_version TEXT,
  error         TEXT
);

CREATE TABLE IF NOT EXISTS assets_seen (
  uuid      TEXT PRIMARY KEY,
  added_at  TEXT NOT NULL,
  run_id    TEXT NOT NULL REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS clusters (
  id         TEXT PRIMARY KEY,
  run_id     TEXT NOT NULL REFERENCES runs(id),
  start_ts   TEXT NOT NULL,
  end_ts     TEXT NOT NULL,
  place      TEXT,
  lat        REAL,
  lon        REAL,
  asset_count INTEGER NOT NULL,
  decision   TEXT,                  -- kept|discarded|unreviewed
  reason     TEXT
);

CREATE TABLE IF NOT EXISTS memories (
  id           TEXT PRIMARY KEY,
  cluster_id   TEXT NOT NULL REFERENCES clusters(id),
  title        TEXT NOT NULL,
  body         TEXT NOT NULL,
  category     TEXT NOT NULL,
  confidence   REAL NOT NULL,
  event_id     TEXT,                -- Google Calendar event id, for undo
  created_at   TEXT NOT NULL
);
