import asyncio
import yfinance as yf
import requests
import json
import time
from datetime import datetime
from typing import List, Dict
import logging
from ..data_processing.data_models import DataPoint

class FinancialDataScraper:
    def __init__(self, alpha_vantage_key: str, fred_key: str):
        self.alpha_vantage_key = alpha_vantage_key
        self.fred_key = fred_key
    
    async def scrape_yahoo_finance(self, tickers: List[str]) -> List[DataPoint]:
        data_points = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Recent news
                news = stock.news
                for article in news[:5]:  # Limit to 5 recent articles
                    data_point = DataPoint(
                        source_type='yahoo_finance',
                        source_name='yahoo_finance',
                        company_ticker=ticker,
                        company_name=info.get('longName'),
                        content_type='financial_news',
                        language='en',
                        title=article.get('title'),
                        content=article.get('summary', ''),
                        url=article.get('link'),
                        published_date=datetime.fromtimestamp(article.get('providerPublishTime', time.time())),
                        metadata={'publisher': article.get('publisher')}
                    )
                    data_points.append(data_point)
                
                # Financial data
                financials_data = {
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'revenue': info.get('totalRevenue'),
                    'profit_margin': info.get('profitMargins')
                }
                
                data_point = DataPoint(
                    source_type='yahoo_finance',
                    source_name='yahoo_finance',
                    company_ticker=ticker,
                    company_name=info.get('longName'),
                    content_type='financial_metrics',
                    language='en',
                    title=f"{ticker} Financial Metrics",
                    content=json.dumps(financials_data),
                    url=None,
                    published_date=datetime.utcnow(),
                    metadata=financials_data
                )
                data_points.append(data_point)
            
            except Exception as e:
                logger.error(f"Error scraping Yahoo Finance for {ticker}: {e}")
        
        return data_points
    
    async def scrape_alpha_vantage(self, tickers: List[str]) -> List[DataPoint]:
        if not self.alpha_vantage_key:
            return []
        
        data_points = []
        
        for ticker in tickers:
            try:
                # Company overview
                url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={self.alpha_vantage_key}"
                response = requests.get(url)
                data = response.json()
                
                if 'Symbol' in data:
                    data_point = DataPoint(
                        source_type='alpha_vantage',
                        source_name='alpha_vantage',
                        company_ticker=ticker,
                        company_name=data.get('Name'),
                        content_type='company_overview',
                        language='en',
                        title=f"{ticker} Company Overview",
                        content=json.dumps(data),
                        url=None,
                        published_date=datetime.utcnow(),
                        metadata=data
                    )
                    data_points.append(data_point)
            
            except Exception as e:
                logger.error(f"Error scraping Alpha Vantage for {ticker}: {e}")
        
        return data_points
