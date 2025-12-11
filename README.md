# Real-Time Explainable Credit Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)

A comprehensive real-time credit risk analysis and monitoring system featuring **event-driven scoring**, **unstructured data integration**, and **Explainable AI (XAI)**. This platform ingests data from over 15 sources to provide immediate, transparent credit intelligence.

## 📖 Overview

Traditional credit scoring is often static and opaque. This platform bridges the gap by utilizing real-time data streams—from financial news to regulatory filings—to dynamically adjust credit scores. It empowers analysts with transparent, AI-driven explanations for every score change, ensuring that risk assessments are both timely and understandable.

## 🚀 Key Features

* **Real-Time Event-Driven Scoring**: Updates credit scores in sub-30 seconds based on detected events (e.g., debt restructuring, earnings warnings).
* **Multi-Source Data Ingestion**: Aggregates data from 15+ sources including NewsAPI, Twitter, Reddit, SEC EDGAR, and market feeds.
* **Unstructured Data Processing**: Uses NLP for entity recognition, sentiment analysis, and event detection on unstructured text.
* **Explainable AI (XAI)**: Provides feature-level insights (SHAP/LIME) and natural language explanations for every score fluctuation.
* **Interactive Analyst Dashboard**: A full-featured frontend for monitoring scores, viewing alerts, and querying the AI assistant.
* **Production-Ready Deployment**: Built on FastAPI with WebSocket support, fully containerized with Docker.

## 🏗️ Architecture Overview

The platform consists of 5 integrated stages plus a comprehensive FastAPI backend:

```mermaid
graph TD
    A[Stage 1: Data Ingestion] -->|Raw Stream| B[Stage 2: Feature Engineering]
    B -->|Feature Vectors| C[Stage 3: ML Scoring]
    C -->|Score & Impact| D[Stage 4: Explainability]
    D -->|Contextualized Data| E[Stage 5: Alerting]
    E -->|Notifications| F[FastAPI Backend]
    F <-->|WebSockets| G[Analyst Dashboard]

```
### Module Breakdown

#### Stage 1: Real-Time Data Ingestion (`stage1_data_ingestion/`)
- **15+ Data Sources**: NewsAPI, Twitter, Reddit, Yahoo Finance, Alpha Vantage, FRED, SEC EDGAR, RSS feeds.
- **Event Detection**: Real-time NLP processing to detect credit-impacting events.
- **Streaming Processing**: Sub-second event processing with priority queues and rate limiting.

#### Stage 2: Feature Engineering (`stage2_feature_engineering/`)
- **Event-Driven Features**: Real-time feature updates based on detected events.
- **NLP Processing**: Advanced sentiment analysis, entity extraction, language detection.
- **Feature Store**: Time-series aggregation with real-time updates and validation.

#### Stage 3: Real-Time Scoring (`stage3_model_training/`)
- **Event-Driven Scoring**: Immediate score updates based on detected events (30-second latency).
- **Impact Calculation**: Calibrated event impact weights (-25 to +5 points) with confidence scoring.
- **Ensemble Models**: XGBoost and LightGBM with real-time inference capabilities.

#### Stage 4: Explainable AI (`stage4_explainability/`)
- **Event Explanations**: Real-time explanations for score changes with event context.
- **Feature-Level Insights**: SHAP and LIME explanations for model transparency.
- **AI Chat Interface**: Natural language explanations of score changes and trends.

#### Stage 5: Real-Time Alerting (`stage5_alerting_workflows/`)
- **Event-Triggered Alerts**: Immediate notifications for critical events (bankruptcy, downgrades).
- **Multi-Channel Notifications**: Email, Slack, Teams, SMS with priority routing.
- **Live Dashboard Feed**: Real-time event stream via WebSockets.

## 🛠️ Tech Stack

- **Language**: Python 3.11+
- **API Framework**: FastAPI
- **Database**: PostgreSQL 15+ (AsyncPG)
- **Caching/Queuing**: Redis 7+
- **ML/AI**: Scikit-learn, XGBoost, SHAP, OpenAI API
- **Containerization**: Docker, Docker Compose

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional)

### 1. Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/credit-intelligence-platform.git](https://github.com/yourusername/credit-intelligence-platform.git)
cd credit-intelligence-platform

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and database settings

# Setup database
createdb credit_intelligence
```

### 2. Run Pipeline

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
```
Access API documentation at: http://localhost:8000/docs

### 4. Docker Deployment
```bash
# Complete system deployment
docker-compose up -d
```

## 🔌 API Endpoints

### Companies & Scoring
- `GET /api/v1/companies`: List companies with advanced filtering.
- `GET /api/v1/companies/{id}`: Get detailed company information.
- `GET /api/v1/scores/company/{id}/current`: Current credit score.
- `GET /api/v1/scores/company/{id}/explanation`: AI explanations.

### Alert Management
- `GET /api/v1/alerts/feed`: Real-time alert feed.
- `POST /api/v1/alerts/{id}/acknowledge`: Acknowledge alert.
- `POST /api/v1/alerts/{id}/resolve`: Resolve alert with notes.

### AI Chat Assistant
- `POST /api/v1/chat/message`: Send message to AI assistant.
- `POST /api/v1/chat/explain`: Get detailed explanations.

### Real-time WebSockets
- `ws://localhost:8000/api/v1/ws/alerts`: Live alert notifications.
- `ws://localhost:8000/api/v1/ws/scores`: Real-time score updates.

## ⚙️ Configuration
### Environment Variables (.env)
```bash
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

# Security
SECRET_KEY=your-secret-key-change-in-production
DEBUG=false
```
### Pipeline Configuration (`config/config.json`)

The `config.json` file controls the behavior of the analysis pipeline:

- **Stage 1**: Data source intervals and API configurations.
- **Stage 2**: NLP models and feature engineering parameters.
- **Stage 3**: ML model settings and training parameters.
- **Stage 4**: Explainability methods and chat configurations.
- **Stage 5**: Alert thresholds and notification channels.

## 📈 Monitoring & Observability

- **Prometheus metrics**: Performance monitoring exposed at `/metrics`.
- **Health checks**: System status monitoring at `/health`.
- **Logging**: Comprehensive structured logging for audit trails.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request
