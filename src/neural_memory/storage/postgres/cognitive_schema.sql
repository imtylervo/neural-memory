-- Cognitive layer tables for PostgreSQL backend
-- Required by PostgresCognitiveMixin

CREATE TABLE IF NOT EXISTS cognitive_state (
    neuron_id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    evidence_for_count INTEGER NOT NULL DEFAULT 0,
    evidence_against_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'confirmed', 'refuted', 'superseded', 'pending', 'expired')),
    predicted_at TEXT,
    resolved_at TEXT,
    schema_version INTEGER DEFAULT 1,
    parent_schema_id TEXT,
    last_evidence_at TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, neuron_id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_cognitive_confidence ON cognitive_state(brain_id, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_cognitive_status ON cognitive_state(brain_id, status);

CREATE TABLE IF NOT EXISTS hot_index (
    brain_id TEXT NOT NULL,
    slot INTEGER NOT NULL,
    category TEXT NOT NULL,
    neuron_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    confidence REAL,
    score REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, slot),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS knowledge_gaps (
    id TEXT PRIMARY KEY,
    brain_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    detection_source TEXT NOT NULL,
    related_neuron_ids TEXT DEFAULT '[]',
    resolved_at TEXT,
    resolved_by_neuron_id TEXT,
    priority REAL DEFAULT 0.5,
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_gaps_brain ON knowledge_gaps(brain_id, resolved_at);
CREATE INDEX IF NOT EXISTS idx_gaps_priority ON knowledge_gaps(brain_id, priority DESC);
