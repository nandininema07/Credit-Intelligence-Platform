# Credit Intelligence Platform - Feature Mapping

## Overview
This document maps the meaningful financial and sentiment features used in our credit intelligence models, replacing the generic `feature_0` through `feature_49` approach.

## Feature Categories

### 1. Liquidity Ratios
| Feature Name | Description | Formula | Good Range | Credit Impact |
|--------------|-------------|---------|-------------|---------------|
| `liquidity_current_ratio` | Current Assets / Current Liabilities | CA/CL | 1.5-3.0 | Higher = Better |
| `liquidity_quick_ratio` | (Current Assets - Inventory) / Current Liabilities | (CA-Inv)/CL | 1.0-2.0 | Higher = Better |
| `liquidity_cash_ratio` | Cash & Equivalents / Current Liabilities | Cash/CL | 0.2-0.5 | Higher = Better |

### 2. Profitability Ratios
| Feature Name | Description | Formula | Good Range | Credit Impact |
|--------------|-------------|---------|-------------|---------------|
| `profitability_gross_margin` | (Revenue - COGS) / Revenue | (Rev-COGS)/Rev | 0.20-0.40 | Higher = Better |
| `profitability_net_margin` | Net Income / Revenue | NI/Rev | 0.05-0.15 | Higher = Better |
| `profitability_roa` | Net Income / Total Assets | NI/TA | 0.05-0.10 | Higher = Better |
| `profitability_roe` | Net Income / Total Equity | NI/TE | 0.10-0.20 | Higher = Better |

### 3. Leverage Ratios
| Feature Name | Description | Formula | Good Range | Credit Impact |
|--------------|-------------|---------|-------------|---------------|
| `leverage_debt_to_equity` | Total Debt / Total Equity | TD/TE | 0.3-1.0 | Lower = Better |
| `leverage_debt_to_assets` | Total Debt / Total Assets | TD/TA | 0.2-0.5 | Lower = Better |
| `leverage_interest_coverage` | EBIT / Interest Expense | EBIT/Int | >3.0 | Higher = Better |

### 4. Efficiency Ratios
| Feature Name | Description | Formula | Good Range | Credit Impact |
|--------------|-------------|---------|-------------|---------------|
| `efficiency_asset_turnover` | Revenue / Total Assets | Rev/TA | 0.5-1.5 | Higher = Better |
| `efficiency_inventory_turnover` | COGS / Average Inventory | COGS/AvgInv | 4-8 | Higher = Better |
| `efficiency_receivables_turnover` | Revenue / Average Receivables | Rev/AvgRec | 6-12 | Higher = Better |

### 5. Credit-Specific Metrics
| Feature Name | Description | Formula | Good Range | Credit Impact |
|--------------|-------------|---------|-------------|---------------|
| `credit_cash_flow_to_debt` | Operating Cash Flow / Total Debt | OCF/TD | >0.15 | Higher = Better |
| `credit_operating_cash_flow_to_debt` | Operating Cash Flow / Total Debt | OCF/TD | >0.20 | Higher = Better |
| `credit_free_cash_flow_to_debt` | Free Cash Flow / Total Debt | FCF/TD | >0.12 | Higher = Better |
| `credit_net_working_capital_to_assets` | Net Working Capital / Total Assets | NWC/TA | 0.10-0.25 | Higher = Better |

### 6. Market Indicators
| Feature Name | Description | Formula | Good Range | Credit Impact |
|--------------|-------------|---------|-------------|---------------|
| `market_pe_ratio` | Price per Share / Earnings per Share | P/E | 10-20 | Moderate = Better |
| `market_pb_ratio` | Price per Share / Book Value per Share | P/B | 1.0-3.0 | Moderate = Better |
| `market_ev_to_ebitda` | Enterprise Value / EBITDA | EV/EBITDA | 8-16 | Lower = Better |
| `market_beta` | Stock volatility vs market | β | 0.8-1.2 | Lower = Better |

### 7. Sentiment & News Features
| Feature Name | Description | Range | Credit Impact |
|--------------|-------------|-------|---------------|
| `sentiment_avg` | Average sentiment score across news | -1.0 to 1.0 | Higher = Better |
| `sentiment_std` | Sentiment volatility | 0.0 to 1.0 | Lower = Better |
| `positive_sentiment_ratio` | Ratio of positive news | 0.0 to 1.0 | Higher = Better |
| `negative_sentiment_ratio` | Ratio of negative news | 0.0 to 1.0 | Lower = Better |
| `event_count` | Total number of news events | 0+ | Moderate = Better |
| `critical_event_count` | Number of critical events | 0+ | Lower = Better |

### 8. Industry & Sector Features
| Feature Name | Description | Range | Credit Impact |
|--------------|-------------|-------|---------------|
| `industry_risk_score` | Industry-specific risk rating | 0.0 to 1.0 | Lower = Better |
| `sector_volatility` | Sector market volatility | 0.1 to 0.5 | Lower = Better |

### 9. Size & Growth Features
| Feature Name | Description | Range | Credit Impact |
|--------------|-------------|-------|---------------|
| `company_size_log` | Log of market capitalization | 5.0 to 12.0 | Higher = Better |
| `revenue_growth_rate` | Annual revenue growth | -0.5 to 0.5 | Higher = Better |
| `earnings_growth_rate` | Annual earnings growth | -0.5 to 0.5 | Higher = Better |

### 10. Macroeconomic Factors
| Feature Name | Description | Range | Credit Impact |
|--------------|-------------|-------|---------------|
| `interest_rate_environment` | Current interest rates | 0.0 to 0.1 | Lower = Better |
| `gdp_growth_rate` | Economic growth rate | -0.1 to 0.1 | Higher = Better |
| `inflation_rate` | Inflation rate | 0.0 to 0.1 | Lower = Better |

## Feature Engineering Pipeline

### Stage 1: Data Collection
- Financial statements (quarterly/annual)
- News and social media feeds
- Market data (stock prices, volumes)
- Regulatory filings (SEC, etc.)

### Stage 2: Feature Engineering
1. **Financial Ratios**: Calculated from raw financial data
2. **Sentiment Analysis**: NLP processing of text data
3. **Market Indicators**: Derived from stock market data
4. **Credit Metrics**: Specialized ratios for credit analysis

### Stage 3: Feature Selection
- Correlation analysis
- Feature importance ranking
- Domain expert validation
- Statistical significance testing

## Model Performance Targets

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Accuracy | >80% | 36-37% | Critical |
| Precision | >75% | 34% | Critical |
| Recall | >70% | 34-35% | Critical |
| F1-Score | >72% | 33-34% | Critical |

## Next Steps

1. **Immediate**: Retrain models with meaningful features
2. **Short-term**: Collect more quality training data
3. **Medium-term**: Implement feature store with real-time updates
4. **Long-term**: Continuous learning and A/B testing

## Feature Validation

Each feature should be validated for:
- **Data Quality**: Missing values, outliers, consistency
- **Business Logic**: Financial sense, domain expertise
- **Statistical Properties**: Distribution, correlation, stability
- **Temporal Stability**: Consistency over time periods

## Monitoring & Maintenance

- **Feature Drift**: Monitor feature distributions over time
- **Performance Degradation**: Track model performance metrics
- **Data Quality**: Automated checks for data integrity
- **Business Rules**: Validation of financial logic
