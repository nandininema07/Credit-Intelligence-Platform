-- Credit Intelligence Platform Database Schema
-- PostgreSQL initialization script

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Companies table
CREATE TABLE IF NOT EXISTS companies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    country VARCHAR(50) DEFAULT 'US',
    exchange VARCHAR(20),
    description TEXT,
    website VARCHAR(255),
    employees INTEGER,
    founded_year INTEGER,
    headquarters VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Raw data points table for ingested data
CREATE TABLE IF NOT EXISTS raw_data_points (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_type VARCHAR(50) NOT NULL,
    source_name VARCHAR(100) NOT NULL,
    company_ticker VARCHAR(10),
    company_name VARCHAR(255),
    content_type VARCHAR(50),
    language VARCHAR(10),
    title TEXT,
    content TEXT,
    url TEXT,
    published_date TIMESTAMP WITH TIME ZONE,
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    sentiment_score DECIMAL(3,2),
    metadata JSONB,
    processed BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (company_ticker) REFERENCES companies(ticker) ON DELETE SET NULL
);

-- Credit events table for detected events
CREATE TABLE IF NOT EXISTS credit_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id VARCHAR(255) UNIQUE NOT NULL,
    company_ticker VARCHAR(10) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    source VARCHAR(100),
    url TEXT,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    published_at TIMESTAMP WITH TIME ZONE,
    confidence_score DECIMAL(3,2),
    impact_direction VARCHAR(20),
    estimated_score_impact DECIMAL(5,2),
    keywords_matched TEXT[],
    metadata JSONB,
    processed BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (company_ticker) REFERENCES companies(ticker) ON DELETE CASCADE
);

-- Credit scores table
CREATE TABLE IF NOT EXISTS credit_scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_ticker VARCHAR(10) NOT NULL,
    score DECIMAL(5,2) NOT NULL,
    risk_category VARCHAR(20),
    probability_default DECIMAL(5,4),
    confidence_score DECIMAL(3,2),
    model_used VARCHAR(50),
    feature_contributions JSONB,
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_current BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (company_ticker) REFERENCES companies(ticker) ON DELETE CASCADE
);

-- Score history for tracking changes
CREATE TABLE IF NOT EXISTS score_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_ticker VARCHAR(10) NOT NULL,
    old_score DECIMAL(5,2),
    new_score DECIMAL(5,2),
    score_change DECIMAL(5,2),
    change_reason TEXT,
    event_id VARCHAR(255),
    model_used VARCHAR(50),
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,
    FOREIGN KEY (company_ticker) REFERENCES companies(ticker) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES credit_events(event_id) ON DELETE SET NULL
);

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_ticker VARCHAR(10) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT,
    threshold_value DECIMAL(10,2),
    current_value DECIMAL(10,2),
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB,
    FOREIGN KEY (company_ticker) REFERENCES companies(ticker) ON DELETE CASCADE
);

-- Features table for ML features
CREATE TABLE IF NOT EXISTS features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_ticker VARCHAR(10) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value DECIMAL(15,6),
    feature_type VARCHAR(50),
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,
    FOREIGN KEY (company_ticker) REFERENCES companies(ticker) ON DELETE CASCADE
);

-- Chat sessions table
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(100),
    company_ticker VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,
    FOREIGN KEY (company_ticker) REFERENCES companies(ticker) ON DELETE SET NULL
);

-- Chat messages table
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL,
    message_type VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
);

-- Model metadata table
CREATE TABLE IF NOT EXISTS model_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50),
    training_date TIMESTAMP WITH TIME ZONE,
    performance_metrics JSONB,
    feature_importance JSONB,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Pipeline runs table for tracking execution
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id VARCHAR(255) UNIQUE NOT NULL,
    stage VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metrics JSONB,
    metadata JSONB
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies(ticker);
CREATE INDEX IF NOT EXISTS idx_companies_sector ON companies(sector);
CREATE INDEX IF NOT EXISTS idx_raw_data_company ON raw_data_points(company_ticker);
CREATE INDEX IF NOT EXISTS idx_raw_data_source ON raw_data_points(source_type, source_name);
CREATE INDEX IF NOT EXISTS idx_raw_data_published ON raw_data_points(published_date);
CREATE INDEX IF NOT EXISTS idx_raw_data_processed ON raw_data_points(processed);
CREATE INDEX IF NOT EXISTS idx_events_company ON credit_events(company_ticker);
CREATE INDEX IF NOT EXISTS idx_events_type ON credit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_detected ON credit_events(detected_at);
CREATE INDEX IF NOT EXISTS idx_events_processed ON credit_events(processed);
CREATE INDEX IF NOT EXISTS idx_scores_company ON credit_scores(company_ticker);
CREATE INDEX IF NOT EXISTS idx_scores_current ON credit_scores(is_current);
CREATE INDEX IF NOT EXISTS idx_scores_calculated ON credit_scores(calculated_at);
CREATE INDEX IF NOT EXISTS idx_score_history_company ON score_history(company_ticker);
CREATE INDEX IF NOT EXISTS idx_score_history_changed ON score_history(changed_at);
CREATE INDEX IF NOT EXISTS idx_alerts_company ON alerts(company_ticker);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_triggered ON alerts(triggered_at);
CREATE INDEX IF NOT EXISTS idx_features_company ON features(company_ticker);
CREATE INDEX IF NOT EXISTS idx_features_name ON features(feature_name);
CREATE INDEX IF NOT EXISTS idx_features_calculated ON features(calculated_at);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_raw_data_content_gin ON raw_data_points USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_events_title_gin ON credit_events USING gin(to_tsvector('english', title));

-- Create functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger 
        WHERE tgname = 'update_companies_updated_at'
    ) THEN
        CREATE TRIGGER update_companies_updated_at
        BEFORE UPDATE ON companies
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    END IF;
END;
$$;


-- Insert default companies for testing
INSERT INTO companies (ticker, name, sector, industry, market_cap, exchange) VALUES
('AAPL', 'Apple Inc.', 'Technology', 'Consumer Electronics', 3000000000000, 'NASDAQ'),
('MSFT', 'Microsoft Corporation', 'Technology', 'Software', 2800000000000, 'NASDAQ'),
('GOOGL', 'Alphabet Inc.', 'Technology', 'Internet Services', 1800000000000, 'NASDAQ'),
('AMZN', 'Amazon.com Inc.', 'Consumer Discretionary', 'E-commerce', 1500000000000, 'NASDAQ'),
('TSLA', 'Tesla Inc.', 'Consumer Discretionary', 'Electric Vehicles', 800000000000, 'NASDAQ'),
('META', 'Meta Platforms Inc.', 'Technology', 'Social Media', 900000000000, 'NASDAQ'),
('NVDA', 'NVIDIA Corporation', 'Technology', 'Semiconductors', 1200000000000, 'NASDAQ'),
('JPM', 'JPMorgan Chase & Co.', 'Financials', 'Banking', 500000000000, 'NYSE'),
('JNJ', 'Johnson & Johnson', 'Healthcare', 'Pharmaceuticals', 450000000000, 'NYSE'),
('V', 'Visa Inc.', 'Financials', 'Payment Processing', 500000000000, 'NYSE'),
('PG', 'Procter & Gamble Co.', 'Consumer Staples', 'Consumer Goods', 380000000000, 'NYSE'),
('UNH', 'UnitedHealth Group Inc.', 'Healthcare', 'Health Insurance', 480000000000, 'NYSE'),
('HD', 'The Home Depot Inc.', 'Consumer Discretionary', 'Retail', 400000000000, 'NYSE'),
('MA', 'Mastercard Inc.', 'Financials', 'Payment Processing', 380000000000, 'NYSE'),
('BAC', 'Bank of America Corp.', 'Financials', 'Banking', 320000000000, 'NYSE'),
('ABBV', 'AbbVie Inc.', 'Healthcare', 'Pharmaceuticals', 300000000000, 'NYSE'),
('PFE', 'Pfizer Inc.', 'Healthcare', 'Pharmaceuticals', 280000000000, 'NYSE'),
('KO', 'The Coca-Cola Company', 'Consumer Staples', 'Beverages', 260000000000, 'NYSE'),
('AVGO', 'Broadcom Inc.', 'Technology', 'Semiconductors', 350000000000, 'NASDAQ'),
('PEP', 'PepsiCo Inc.', 'Consumer Staples', 'Beverages', 240000000000, 'NASDAQ')
ON CONFLICT (ticker) DO NOTHING;

-- Create views for common queries
CREATE OR REPLACE VIEW current_scores AS
SELECT 
    cs.*,
    c.name as company_name,
    c.sector,
    c.industry
FROM credit_scores cs
JOIN companies c ON cs.company_ticker = c.ticker
WHERE cs.is_current = TRUE;

CREATE OR REPLACE VIEW recent_events AS
SELECT 
    ce.*,
    c.name as company_name,
    c.sector
FROM credit_events ce
JOIN companies c ON ce.company_ticker = c.ticker
WHERE ce.detected_at >= NOW() - INTERVAL '7 days'
ORDER BY ce.detected_at DESC;

CREATE OR REPLACE VIEW active_alerts AS
SELECT 
    a.*,
    c.name as company_name,
    c.sector
FROM alerts a
JOIN companies c ON a.company_ticker = c.ticker
WHERE a.status = 'active'
ORDER BY a.triggered_at DESC;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO credit_intelligence_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO credit_intelligence_user;

COMMIT;
