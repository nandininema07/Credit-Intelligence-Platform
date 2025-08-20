# Database Setup Guide

This directory contains database setup scripts and configuration for the Credit Intelligence Platform.

## Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Run the automated setup script
python database/setup_db.py
```

This script will:
- ✅ Create the `credit_intelligence` database
- ✅ Run all initialization scripts
- ✅ Create tables, indexes, and views
- ✅ Insert sample company data
- ✅ Test connections and verify setup

### Option 2: Manual Setup

1. **Install PostgreSQL** (if not already installed):
   ```bash
   # Windows (using chocolatey)
   choco install postgresql
   
   # Or download from: https://www.postgresql.org/download/
   ```

2. **Start PostgreSQL service**:
   ```bash
   # Windows
   net start postgresql-x64-15
   
   # Or use pgAdmin or Services panel
   ```

3. **Create database manually**:
   ```sql
   -- Connect to PostgreSQL as postgres user
   psql -U postgres
   
   -- Create database
   CREATE DATABASE credit_intelligence;
   
   -- Exit
   \q
   ```

4. **Run initialization script**:
   ```bash
   # Run the SQL script
   psql -U postgres -d credit_intelligence -f database/init_db.sql
   ```

## Database Schema

### Core Tables

- **`companies`** - Company master data (ticker, name, sector, etc.)
- **`raw_data_points`** - Ingested data from all sources
- **`credit_events`** - Detected credit-impacting events
- **`credit_scores`** - Current and historical credit scores
- **`score_history`** - Score change tracking with reasons
- **`alerts`** - Generated alerts and notifications
- **`features`** - ML features for scoring models

### Supporting Tables

- **`chat_sessions`** / **`chat_messages`** - AI chat interface data
- **`model_metadata`** - ML model versioning and performance
- **`pipeline_runs`** - Pipeline execution tracking

### Views

- **`current_scores`** - Latest scores with company info
- **`recent_events`** - Events from last 7 days
- **`active_alerts`** - Unresolved alerts

## Configuration

Update your `.env` file with database connection details:

```env
# Database Configuration
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost/credit_intelligence
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=password
DB_NAME=credit_intelligence
```

## Sample Data

The initialization script includes 20 major companies:
- AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA
- JPM, JNJ, V, PG, UNH, HD, MA, BAC
- ABBV, PFE, KO, AVGO, PEP

## Troubleshooting

### Common Issues

1. **PostgreSQL not running**:
   ```bash
   # Check service status
   sc query postgresql-x64-15
   
   # Start service
   net start postgresql-x64-15
   ```

2. **Connection refused**:
   - Verify PostgreSQL is running on port 5432
   - Check firewall settings
   - Verify username/password in `.env`

3. **Permission denied**:
   ```sql
   -- Grant permissions to user
   GRANT ALL PRIVILEGES ON DATABASE credit_intelligence TO your_user;
   ```

4. **Database already exists**:
   ```sql
   -- Drop and recreate if needed
   DROP DATABASE IF EXISTS credit_intelligence;
   CREATE DATABASE credit_intelligence;
   ```

### Verification

Test your database setup:

```bash
# Test connection
python -c "
import asyncpg
import asyncio

async def test():
    conn = await asyncpg.connect('postgresql://postgres:password@localhost/credit_intelligence')
    result = await conn.fetchval('SELECT COUNT(*) FROM companies')
    print(f'✅ Connected! Found {result} companies')
    await conn.close()

asyncio.run(test())
"
```

## Performance Optimization

The schema includes optimized indexes for:
- Company lookups by ticker
- Time-series queries on scores and events
- Full-text search on content
- Real-time alert queries

## Backup & Restore

```bash
# Backup
pg_dump -U postgres credit_intelligence > backup.sql

# Restore
psql -U postgres -d credit_intelligence < backup.sql
```
