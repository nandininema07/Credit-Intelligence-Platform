# Credit Intelligence Platform - Backend API

A comprehensive FastAPI backend for the Credit Intelligence Platform that integrates all 5 pipeline stages for real-time credit risk analysis and monitoring.

## Features

- **Stage 1**: Data ingestion from financial, news, and regulatory sources
- **Stage 2**: Feature engineering and data processing
- **Stage 3**: ML model training and scoring
- **Stage 4**: AI explainability and chatbot functionality
- **Stage 5**: Real-time alerting and workflow automation

## Architecture

```
├── app/
│   ├── api/                 # API route handlers
│   │   ├── companies.py     # Company management endpoints
│   │   ├── scores.py        # Credit scoring endpoints
│   │   ├── alerts.py        # Alert management endpoints
│   │   ├── chat.py          # AI chatbot endpoints
│   │   └── websocket.py     # Real-time WebSocket endpoints
│   ├── models/              # Pydantic data models
│   ├── services/            # Business logic services
│   ├── middleware/          # Custom middleware
│   ├── utils/               # Utility functions
│   ├── config.py            # Configuration management
│   └── database.py          # Database connection
├── main.py                  # FastAPI application entry point
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
└── docker-compose.yml     # Multi-service orchestration
```

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Credit-Intelligence-Platform/backend
```

2. **Set up environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

1. **Build and run with Docker Compose**
```bash
docker-compose up -d
```

2. **Access the application**
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Grafana: http://localhost:3001 (admin/admin)
- Prometheus: http://localhost:9090

## API Endpoints

### Companies
- `GET /api/v1/companies` - List companies with filtering
- `GET /api/v1/companies/{id}` - Get company details
- `GET /api/v1/companies/search` - Search companies
- `POST /api/v1/companies` - Create company
- `PUT /api/v1/companies/{id}` - Update company

### Credit Scores
- `GET /api/v1/scores/company/{id}/current` - Current score
- `GET /api/v1/scores/company/{id}/history` - Score history
- `GET /api/v1/scores/company/{id}/prediction` - Score prediction
- `GET /api/v1/scores/company/{id}/explanation` - AI explanation
- `GET /api/v1/scores/top-movers` - Top score changes

### Alerts
- `GET /api/v1/alerts` - List alerts with filtering
- `GET /api/v1/alerts/feed` - Real-time alert feed
- `POST /api/v1/alerts/{id}/acknowledge` - Acknowledge alert
- `POST /api/v1/alerts/{id}/resolve` - Resolve alert
- `POST /api/v1/alerts/{id}/share` - Share/export alert

### AI Chat
- `POST /api/v1/chat/message` - Send chat message
- `GET /api/v1/chat/suggestions` - Get question suggestions
- `POST /api/v1/chat/explain` - Get detailed explanations

### WebSocket
- `ws://localhost:8000/api/v1/ws/alerts` - Real-time alerts
- `ws://localhost:8000/api/v1/ws/scores` - Score updates
- `ws://localhost:8000/api/v1/ws/dashboard` - Dashboard updates

## Configuration

### Environment Variables

Key configuration options in `.env`:

```env
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/db

# Security
SECRET_KEY=your-secret-key
DEBUG=false

# External APIs
OPENAI_API_KEY=your-openai-key
SLACK_BOT_TOKEN=your-slack-token
JIRA_API_TOKEN=your-jira-token

# Email
SMTP_HOST=smtp.gmail.com
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

### Stage-Specific Configuration

Each pipeline stage has dedicated configuration in `config.py`:

- **Stage 1**: Data source configurations, update frequencies
- **Stage 2**: Feature engineering parameters, aggregation windows
- **Stage 3**: Model parameters, training configurations
- **Stage 4**: AI model settings, explainability methods
- **Stage 5**: Alert thresholds, notification channels

## Development

### Running Tests
```bash
pytest app/tests/ -v --cov=app
```

### Code Quality
```bash
# Format code
black app/
isort app/

# Lint code
flake8 app/
mypy app/
```

### Database Migrations
```bash
# Generate migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head
```

## Monitoring

### Health Checks
- `GET /health` - Application health status
- `GET /api/v1/system/stats` - System statistics

### Metrics
- Prometheus metrics at `:9090`
- Grafana dashboards at `:3001`
- Application logs in `/app/logs`

## Security

- JWT-based authentication
- Role-based access control
- Rate limiting on API endpoints
- Input validation and sanitization
- HTTPS support with SSL termination
- Security headers via Nginx

## Integration

### Frontend Integration
The backend provides all necessary endpoints for the Next.js frontend:

- Company data and search
- Real-time score updates
- Alert management
- AI chat functionality
- WebSocket connections for live updates

### External Services
- **Slack**: Alert notifications and bot integration
- **Microsoft Teams**: Team collaboration alerts
- **Jira**: Automated ticket creation from alerts
- **Email**: SMTP-based notifications
- **SMS**: Twilio integration for critical alerts

## Performance

- Async/await throughout for non-blocking operations
- Connection pooling for database and Redis
- Caching with Redis for frequently accessed data
- Pagination for large datasets
- Background tasks for heavy processing
- Load balancing with Nginx

## Deployment

### Production Checklist

- [ ] Set `DEBUG=false`
- [ ] Configure strong `SECRET_KEY`
- [ ] Set up SSL certificates
- [ ] Configure database backups
- [ ] Set up monitoring and alerting
- [ ] Configure log rotation
- [ ] Set resource limits
- [ ] Enable security headers

### Scaling

- Horizontal scaling with multiple backend instances
- Database read replicas for query performance
- Redis clustering for cache scaling
- CDN for static assets
- Load balancer health checks

## Troubleshooting

### Common Issues

1. **Database Connection**: Check `DATABASE_URL` and PostgreSQL status
2. **Redis Connection**: Verify Redis is running and accessible
3. **WebSocket Issues**: Check Nginx WebSocket proxy configuration
4. **Authentication**: Verify JWT token format and expiration
5. **External APIs**: Check API keys and network connectivity

### Logs

Application logs are structured and include:
- Request/response logging
- Error tracking with stack traces
- Performance metrics
- Security events

## Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review application logs
3. Check system health at `/health`
4. Monitor metrics in Grafana

## License

Copyright © 2024 Credit Intelligence Platform. All rights reserved.
