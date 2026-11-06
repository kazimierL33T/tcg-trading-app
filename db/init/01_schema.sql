-- ==============================================================================================
-- Canadian based TCG platform - PostgreSQL schema
-- Requires PostgreSQL 15+ and the pgvector extension for the scanner.
-- Runs automatically the first time the db container starts on an empty volume (see docker-compose.yml).
-- ==============================================================================================

-- Enable pgvector so we can store card embeddings for the scanner.
CREATE EXTENSION IF NOT EXISTS vector;
-- Enable trigram matching for fast fuzzy card-name search.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Standard card variations.
-- Using enumeration labels here. Values are case-sensitive: always use them exactly as written.
CREATE TYPE card_condition AS ENUM ('NM', 'LP', 'MP', 'HP', 'DMG');

CREATE TYPE card_finish AS ENUM ('Normal', 'Foil', 'Rainbow-Foil', 'Showcase', 'Etched', 'Textured', 'Holo', 'Reverse-Holo', 'Other');

CREATE TYPE listing_status AS ENUM ('Active', 'Sold', 'Cancelled');

-- Where the price decision came from.
CREATE TYPE price_source AS ENUM ('TCGPlayer', 'Current Market', 'Past Sale');

-- --------------------------------------------------------------------------------------------
-- CATALOG: games, sets, cards, printings
-- --------------------------------------------------------------------------------------------

CREATE TABLE games (
    id SMALLSERIAL PRIMARY KEY,
    code TEXT NOT NULL UNIQUE, -- Short lowercase code for the game, e.g. 'mtg'.
    name TEXT NOT NULL UNIQUE
);

-- Sets/expansions
CREATE TABLE sets (
    id SERIAL PRIMARY KEY,
    game_id SMALLINT NOT NULL REFERENCES games(id),
    external_id TEXT NOT NULL, -- ID from the external API we import data from.
    code TEXT NOT NULL,
    name TEXT NOT NULL,
    release_date DATE,
    card_count INT,
    UNIQUE (game_id, external_id)
);

-- Game-level card data; game-specific fields live in JSONB
CREATE TABLE cards (
    id SERIAL PRIMARY KEY,
    game_id SMALLINT NOT NULL REFERENCES games(id),
    external_id TEXT NOT NULL,
    name TEXT NOT NULL,
    card_type TEXT,
    rules_text TEXT,
    attributes JSONB NOT NULL DEFAULT '{}', -- mana, colors, keywords, etc
    UNIQUE (game_id, external_id)
);

-- Trigram index for fuzzy named card searching
CREATE INDEX idx_cards_name_trgm ON cards USING gin (name gin_trgm_ops);

-- Specific printings of a card
CREATE TABLE printings (
    id SERIAL PRIMARY KEY,
    card_id INT NOT NULL REFERENCES cards(id),
    set_id INT NOT NULL REFERENCES sets(id),
    external_id TEXT NOT NULL,
    collector_number TEXT NOT NULL,
    rarity TEXT,
    artist TEXT,
    image_url TEXT,  -- remote source image
    image_path TEXT, -- local copy used for ML training
    finishes card_finish[] NOT NULL DEFAULT '{Normal}',
    tcgplayer_id TEXT,
    UNIQUE (set_id, external_id)
);

-- Speeds up "all printings of a card" queries.
CREATE INDEX idx_printings_card_id ON printings(card_id);

-- --------------------------------------------------------------------------------------------
-- USERS, COLLECTIONS, LISTINGS
-- --------------------------------------------------------------------------------------------

-- Accounts; store only a password hash (argon2/bcrypt)... never store plaintext passwords.
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL,
    province CHAR(2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- What a user owns; one row per printing/condition/finish combination.
CREATE TABLE collection_items (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    printing_id INT NOT NULL REFERENCES printings(id),
    condition card_condition NOT NULL DEFAULT 'NM',
    finish card_finish NOT NULL DEFAULT 'Normal',
    quantity INT NOT NULL CHECK (quantity > 0),
    acquired_price_cad NUMERIC (10, 2),
    added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, printing_id, condition, finish)
);

-- Cards for sale. Prices stored in CAD, but we can add other currencies later if needed.
CREATE TABLE listings (
    id SERIAL PRIMARY KEY,
    seller_id INTEGER NOT NULL REFERENCES users(id),
    printing_id INTEGER NOT NULL REFERENCES printings(id),
    condition card_condition NOT NULL DEFAULT 'NM',
    finish card_finish NOT NULL DEFAULT 'Normal',
    quantity INT NOT NULL CHECK (quantity > 0),
    price_cad NUMERIC (10, 2) NOT NULL CHECK (price_cad > 0),
    status listing_status NOT NULL DEFAULT 'Active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- fast 'Cheapest active listings for this printing' queries
CREATE INDEX idx_listings_active ON listings(printing_id, price_cad) WHERE status = 'Active';

-- --------------------------------------------------------------------------------------------
-- PRICING
-- --------------------------------------------------------------------------------------------

-- Append-only price history, feeds charts and later price predictions
CREATE TABLE price_history (
    id BIGSERIAL PRIMARY KEY,
    printing_id INT NOT NULL REFERENCES printings(id),
    finish card_finish NOT NULL DEFAULT 'Normal',
    condition card_condition NOT NULL DEFAULT 'NM',
    price_cad NUMERIC (10, 2) NOT NULL,
    source price_source NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Price chart queries filter by printing and sort by time
CREATE INDEX idx_price_history_lookup ON price_history(printing_id, finish, recorded_at DESC);

-- Daily USD->CAD rate so USD prices convert automatically
CREATE TABLE exchange_rates (
    rate_date DATE PRIMARY KEY,
    usd_cad NUMERIC(8, 5) NOT NULL
);

-- --------------------------------------------------------------------------------------------
-- DATA IMPORT TRACKING
-- --------------------------------------------------------------------------------------------

-- Logs each importer run so failures and row counts are auditable.
CREATE TABLE import_runs (
    id SERIAL PRIMARY KEY,
    game_id SMALLINT NOT NULL REFERENCES games(id),
    source TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'Running', -- 'Running', 'Success' or 'Failed'
    rows_upserted INT DEFAULT 0,
    error TEXT
);

-- --------------------------------------------------------------------------------------------
-- MACHINE LEARNING SCANNER: embedding model + scanner feedback loop
-- --------------------------------------------------------------------------------------------

-- Every model I train gets a row, so I can compare and roll back
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    architecture TEXT NOT NULL,
    embedding_dim INT NOT NULL,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metrics JSONB NOT NULL DEFAULT '{}',
    is_active BOOLEAN NOT NULL DEFAULT FALSE
);

-- Only one model can serve scans at a time
CREATE UNIQUE INDEX idx_one_active_model ON model_versions(is_active) WHERE is_active;

-- One embedding per printing per model; dimension fixed at 256 for now
CREATE TABLE card_embeddings (
    printing_id INT NOT NULL REFERENCES printings(id) ON DELETE CASCADE,
    model_version_id INT NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    embedding vector(256) NOT NULL,
    PRIMARY KEY (printing_id, model_version_id)
);

-- HNSW index for fast nearest-neighbour search by cosine distance
CREATE INDEX idx_card_embeddings_hnsw ON card_embeddings USING hnsw (embedding vector_cosine_ops);

-- Every scan is logged; user-confirmed answers become new training labels
CREATE TABLE scan_events (
    id BIGSERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE SET NULL,
    model_version_id INT NOT NULL REFERENCES model_versions(id),
    image_path TEXT NOT NULL,                              -- stored upload, for retraining
    top_k JSONB NOT NULL,                                  -- [{"printing_id": 1, "distance": 0.12}, ...]
    predicted_printing_id INT REFERENCES printings(id),
    confirmed_printing_id INT REFERENCES printings(id),    -- NULL until the user confirms/corrects
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Pull confirmed scans quickly when building the next training set
CREATE INDEX idx_scan_events_confirmed ON scan_events(confirmed_printing_id) WHERE confirmed_printing_id IS NOT NULL;

-- --------------------------------------------------------------------------------------------
-- SEED DATA
-- --------------------------------------------------------------------------------------------

-- The two MVP games (lowercase codes; the Day 4 tests check for these)
INSERT INTO games (code, name) VALUES
    ('mtg', 'Magic: The Gathering'),
    ('pokemon', 'Pokemon TCG');