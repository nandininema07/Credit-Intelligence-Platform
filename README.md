# Real-Time Explainable Credit Intelligence Platform

A comprehensive real-time credit risk analysis and monitoring system with **event-driven scoring** and **explainable AI** for the IITK CredTech Hackathon.

## 🏆 Hackathon Implementation Status

✅ **Real-Time Event-Driven Scoring**
✅ **Multi-Source Data Ingestion (15+ Sources)**
✅ **Unstructured Event Detection & Integration**
✅ **Explainable AI with Feature-Level Insights**
✅ **Interactive Analyst Dashboard**
✅ **Production-Ready Deployment**

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

### Stage 1: Real-Time Data Ingestion (`stage1_data_ingestion/`)
- **15+ Data Sources**: NewsAPI, Twitter, Reddit, Yahoo Finance, Alpha Vantage, FRED, SEC EDGAR, RSS feeds
- **Event Detection**: Real-time NLP processing to detect credit-impacting events (debt restructuring, earnings warnings, etc.)
- **Streaming Processing**: Sub-second event processing with priority queues and rate limiting
- **Multi-Source Collector**: Async data collection from all sources with fault tolerance

### Stage 2: Feature Engineering (`stage2_feature_engineering/`)
- **Event-Driven Features**: Real-time feature updates based on detected events
- **NLP Processing**: Advanced sentiment analysis, entity extraction, language detection
- **Financial Metrics**: Market indicators, volatility analysis, trend detection
- **Feature Store**: Time-series aggregation with real-time updates and validation

### Stage 3: Real-Time Scoring (`stage3_model_training/`)
- **Event-Driven Scoring**: Immediate score updates based on detected events (30-second latency)
- **Impact Calculation**: Calibrated event impact weights (-25 to +5 points) with confidence scoring
- **Ensemble Models**: XGBoost, LightGBM with real-time inference capabilities
- **Dynamic Risk Assessment**: Continuous score recalculation with time-decay factors

### Stage 4: Explainable AI (`stage4_explainability/`)
- **Event Explanations**: Real-time explanations for score changes with event context
- **Feature-Level Insights**: SHAP and LIME explanations for model transparency
- **AI Chat Interface**: Natural language explanations of score changes and trends
- **Trend Analysis**: Short-term vs long-term indicators with reasoning highlights

### Stage 5: Real-Time Alerting (`stage5_alerting_workflows/`)
- **Event-Triggered Alerts**: Immediate notifications for critical events (bankruptcy, downgrades)
- **Score Change Alerts**: Threshold-based alerts with configurable sensitivity
- **Multi-Channel Notifications**: Email, Slack, Teams, SMS with priority routing
- **Live Dashboard Feed**: Real-time event stream with WebSocket connections

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

## 🚀 Key Hackathon Features

### Real-Time Event-Driven Scoring
- **Sub-30 Second Updates**: Events detected and scored faster than traditional ratings
- **9 Event Types**: Debt restructuring, earnings warnings, regulatory actions, etc.
- **Impact Calibration**: Each event type has calibrated score impact (-25 to +5 points)
- **Confidence Scoring**: Events filtered by confidence thresholds and time decay

### Unstructured Data Integration
- **NLP Event Detection**: Advanced pattern matching for credit-impacting events
- **Multi-Source Processing**: News, social media, filings processed simultaneously
- **Entity Recognition**: Company mentions extracted with alias matching
- **Sentiment Integration**: Event sentiment factored into score calculations

### Explainable AI
- **Event Explanations**: Every score change includes event context and reasoning
- **Feature Contributions**: SHAP-based explanations for model transparency
- **Trend Analysis**: Short-term vs long-term indicators with plain-language summaries
- **Interactive Chat**: AI assistant explains score changes and provides insights

### Production-Ready Architecture
- **Streaming Processing**: Async event queues with priority handling
- **Fault Tolerance**: Graceful handling of data source outages
- **WebSocket Integration**: Real-time dashboard updates
- **Scalable Design**: Handles dozens of companies with rate limiting

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

## 🎯 Hackathon Deliverables

### 1. **Code Repository** ✅
- Complete implementation with atomic commits
- Clear documentation and setup instructions
- Modular architecture with separation of concerns

### 2. **Public Demo URL** 🚧
- FastAPI backend with interactive documentation
- Real-time dashboard with live score updates
- WebSocket connections for event streaming

### 3. **Video Walkthrough** 📹
- Key features demonstration (5-7 minutes)
- End-to-end product flow
- Technical implementation highlights

### 4. **Presentation Deck** 📊
- System architecture and design decisions
- Model performance and explainability
- Innovation highlights and competitive advantages

## 🏅 Evaluation Alignment

- **Data Engineering (20%)**: Multi-source ingestion, streaming processing, fault tolerance
- **Model Accuracy & Explainability (30%)**: Event-driven scoring, SHAP explanations, confidence metrics
- **Unstructured Data (12.5%)**: NLP event detection, sentiment integration, entity recognition
- **User Experience (15%)**: Interactive dashboard, real-time updates, analyst-friendly interface
- **Deployment (10%)**: Production-ready FastAPI, Docker containers, monitoring
- **Innovation (12.5%)**: Real-time event processing, sub-30-second scoring, explainable updates
