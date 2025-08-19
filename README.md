# Credit Intelligence Platform

A comprehensive real-time credit risk analysis and monitoring system with integrated AI/ML pipeline.

## Complete Implementation Status

✅ **All 5 Pipeline Stages Implemented**
✅ **FastAPI Backend with Full API**
✅ **Production-Ready Deployment**
✅ **End-to-End Integration**

## Architecture Overview

The platform consists of 5 integrated stages plus a comprehensive FastAPI backend:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Credit Intelligence Platform                  │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: Data Ingestion → Stage 2: Feature Engineering        │
│                                    ↓                            │
│  Stage 5: Alerting ← Stage 4: Explainability ← Stage 3: ML     │
│                                    ↓                            │
│              FastAPI Backend + Frontend Integration             │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 1: Data Ingestion (`stage1_data_ingestion/`)
- **Multi-source scrapers**: News, social media, financial APIs, regulatory filings
- **Real-time processing**: Language detection, entity extraction, sentiment analysis
- **PostgreSQL storage**: Compressed data with quality monitoring
- **Health monitoring**: Pipeline status and metrics collection

### Stage 2: Feature Engineering (`stage2_feature_engineering/`)
- **NLP processing**: Sentiment analysis, topic modeling, entity linking, event detection
- **Financial analysis**: Ratio calculations, trend analysis, volatility metrics
- **Feature store**: Time series aggregation and feature scaling
- **Market indicators**: Comprehensive market-based feature generation

### Stage 3: Model Training (`stage3_model_training/`)
- **Ensemble models**: XGBoost, LightGBM, Random Forest with hyperparameter tuning
- **Automated training**: Performance monitoring, drift detection, retraining
- **Model versioning**: Metadata tracking and model registry
- **Credit scoring**: Risk categorization and confidence scoring

### Stage 4: Explainability (`stage4_explainability/`)
- **ML explanations**: SHAP and LIME for model transparency
- **AI chat interface**: OpenAI integration for natural language explanations
- **Interactive queries**: Context-aware responses and recommendations
- **Explanation caching**: Performance optimization for repeated queries

### Stage 5: Alerting & Workflows (`stage5_alerting_workflows/`)
- **Real-time alerting**: Score-based and threshold alerts with cooldown management
- **Multi-channel notifications**: Email, Slack, Teams, SMS integration
- **Workflow automation**: Jira ticket creation, PDF report generation
- **Live feed**: Real-time event streaming and alert history

### FastAPI Backend (`backend/`)
- **Complete REST API**: Companies, scores, alerts, chat, WebSocket endpoints
- **Authentication**: JWT-based with role-based access control
- **Real-time features**: WebSocket connections for live updates
- **Production ready**: Docker, monitoring, security, documentation

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional)

### 1. Complete Pipeline Setup

```bash
# Clone and setup
git clone <repository-url>
cd Credit-Intelligence-Platform

# Install all dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and database settings

# Setup database
createdb credit_intelligence
```

### 2. Run Complete Pipeline

```bash
# Run all 5 stages integrated
python run_pipeline.py --mode run

# Test single company processing
python run_pipeline.py --mode test --company "Apple Inc."

# Check pipeline status
python run_pipeline.py --mode status
```

### 3. Run FastAPI Backend

```bash
# Start backend API server
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Access API documentation
# http://localhost:8000/docs
```

### 4. Docker Deployment

```bash
# Complete system deployment
docker-compose up -d

# Backend only
cd backend
docker-compose up -d
```

## API Endpoints

### Companies Management
- `GET /api/v1/companies` - List companies with advanced filtering
- `GET /api/v1/companies/{id}` - Get detailed company information
- `GET /api/v1/companies/search` - Search companies by name/ticker
- `POST /api/v1/companies` - Create new company
- `PUT /api/v1/companies/{id}` - Update company information
- `GET /api/v1/companies/{id}/peers` - Get peer companies
- `GET /api/v1/companies/{id}/risk-factors` - Get risk factor analysis

### Credit Scoring
- `GET /api/v1/scores/company/{id}/current` - Current credit score
- `GET /api/v1/scores/company/{id}/history` - Historical score data
- `GET /api/v1/scores/company/{id}/prediction` - ML-powered predictions
- `GET /api/v1/scores/company/{id}/explanation` - AI explanations
- `GET /api/v1/scores/company/{id}/comparison` - Peer comparisons
- `GET /api/v1/scores/benchmarks` - Industry benchmarks
- `GET /api/v1/scores/top-movers` - Biggest score changes

### Alert Management
- `GET /api/v1/alerts` - List alerts with filtering
- `GET /api/v1/alerts/feed` - Real-time alert feed
- `GET /api/v1/alerts/summary` - Alert summary and metrics
- `POST /api/v1/alerts/{id}/acknowledge` - Acknowledge alert
- `POST /api/v1/alerts/{id}/resolve` - Resolve alert with notes
- `POST /api/v1/alerts/{id}/share` - Share/export alert
- `POST /api/v1/alerts/{id}/create-task` - Create workflow task

### AI Chat Assistant
- `POST /api/v1/chat/message` - Send message to AI assistant
- `GET /api/v1/chat/suggestions` - Get suggested questions
- `POST /api/v1/chat/explain` - Get detailed explanations
- `GET /api/v1/chat/sessions` - Manage chat sessions
- `POST /api/v1/chat/feedback` - Provide feedback on responses

### Real-time WebSocket
- `ws://localhost:8000/api/v1/ws/alerts` - Live alert notifications
- `ws://localhost:8000/api/v1/ws/scores` - Real-time score updates
- `ws://localhost:8000/api/v1/ws/dashboard` - Dashboard live data
- `ws://localhost:8000/api/v1/ws/chat` - Real-time chat interface

## Configuration

### Environment Variables (`.env`)

```env
# Database Configuration
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost/credit_intelligence

# External API Keys
OPENAI_API_KEY=your-openai-api-key
NEWSAPI_KEY=your-newsapi-key
TWITTER_BEARER_TOKEN=your-twitter-token
ALPHA_VANTAGE_KEY=your-alpha-vantage-key

# Notification Services
SMTP_HOST=smtp.gmail.com
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SLACK_BOT_TOKEN=xoxb-your-slack-token
TEAMS_WEBHOOK_URL=your-teams-webhook-url
JIRA_API_TOKEN=your-jira-token

# Security
SECRET_KEY=your-secret-key-change-in-production
DEBUG=false
```

### Pipeline Configuration (`config/config.json`)

Each stage has comprehensive configuration options:
- **Stage 1**: Data source intervals, API configurations
- **Stage 2**: NLP models, feature engineering parameters
- **Stage 3**: ML model settings, training parameters
- **Stage 4**: Explainability methods, chat configurations
- **Stage 5**: Alert thresholds, notification channels

## Key Features

### Real-time Data Processing
- **15+ data sources**: Financial APIs, news, social media, regulatory filings
- **Multi-language support**: Automatic language detection and processing
- **Quality monitoring**: Data validation and health checks
- **Scalable ingestion**: Async processing with rate limiting

- **API key management**: Secure credential storage
- **Data encryption**: Sensitive data protection
- **JWT authentication**: Secure API access
- **Input validation**: Comprehensive data validation

## 📈 Monitoring & Observability

- **Prometheus metrics**: Performance monitoring
- **Health checks**: System status monitoring
- **Logging**: Comprehensive audit trails
- **Alert statistics**: Alert frequency and resolution tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation in `/docs`
- Review the configuration examples

## 🔮 Roadmap

- [ ] Real-time streaming data processing
- [ ] Advanced ensemble models
- [ ] Mobile app integration
- [ ] Blockchain integration for audit trails
- [ ] Advanced visualization dashboard
- [ ] Multi-tenant architecture
