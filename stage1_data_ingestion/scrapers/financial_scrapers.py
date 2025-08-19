"""
Financial data scrapers for Yahoo Finance, Alpha Vantage, and other financial APIs.
Handles real-time and historical financial data collection.
"""

import yfinance as yf
import requests
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import pandas as pd
import json

logger = logging.getLogger(__name__)

@dataclass
class FinancialData:
    """Data class for financial information"""
    symbol: str
    company_name: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    dividend_yield: Optional[float]
    timestamp: datetime
    source: str
    currency: str = 'USD'

@dataclass
class FinancialNews:
    """Data class for financial news"""
    title: str
    summary: str
    url: str
    source: str
    published_date: datetime
    symbols: List[str]
    sentiment: Optional[str] = None

class FinancialScrapers:
    """Financial data scrapers for multiple sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_keys = config.get('api_keys', {})
        self.session = requests.Session()
        
    async def scrape_yahoo_finance(self, symbols: List[str]) -> List[FinancialData]:
        """Scrape Yahoo Finance for stock data"""
        financial_data = []
        
        try:
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    
                    data = FinancialData(
                        symbol=symbol,
                        company_name=info.get('longName', symbol),
                        price=float(latest['Close']),
                        change=float(latest['Close'] - latest['Open']),
                        change_percent=float((latest['Close'] - latest['Open']) / latest['Open'] * 100),
                        volume=int(latest['Volume']),
                        market_cap=info.get('marketCap'),
                        pe_ratio=info.get('trailingPE'),
                        dividend_yield=info.get('dividendYield'),
                        timestamp=datetime.now(),
                        source='Yahoo Finance',
                        currency=info.get('currency', 'USD')
                    )
                    financial_data.append(data)
                    
            logger.info(f"Scraped {len(financial_data)} stocks from Yahoo Finance")
            return financial_data
            
        except Exception as e:
            logger.error(f"Error scraping Yahoo Finance: {e}")
            return []
    
    async def scrape_alpha_vantage(self, symbols: List[str]) -> List[FinancialData]:
        """Scrape Alpha Vantage for financial data"""
        if 'alpha_vantage' not in self.api_keys:
            logger.warning("Alpha Vantage API key not found")
            return []
            
        financial_data = []
        api_key = self.api_keys['alpha_vantage']
        
        try:
            for symbol in symbols:
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': api_key
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        data = await response.json()
                        
                quote = data.get('Global Quote', {})
                if quote:
                    data_point = FinancialData(
                        symbol=symbol,
                        company_name=symbol,  # Alpha Vantage doesn't provide company name in this endpoint
                        price=float(quote.get('05. price', 0)),
                        change=float(quote.get('09. change', 0)),
                        change_percent=float(quote.get('10. change percent', '0%').replace('%', '')),
                        volume=int(quote.get('06. volume', 0)),
                        market_cap=None,
                        pe_ratio=None,
                        dividend_yield=None,
                        timestamp=datetime.now(),
                        source='Alpha Vantage'
                    )
                    financial_data.append(data_point)
                    
                # Rate limiting for free tier
                await asyncio.sleep(12)  # 5 calls per minute limit
                
            logger.info(f"Scraped {len(financial_data)} stocks from Alpha Vantage")
            return financial_data
            
        except Exception as e:
            logger.error(f"Error scraping Alpha Vantage: {e}")
            return []
    
    async def scrape_financial_news(self, symbols: List[str]) -> List[FinancialNews]:
        """Scrape financial news for specific symbols"""
        news_items = []
        
        try:
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                news = ticker.news
                
                for item in news[:5]:  # Top 5 news items per symbol
                    news_item = FinancialNews(
                        title=item.get('title', ''),
                        summary=item.get('summary', ''),
                        url=item.get('link', ''),
                        source=item.get('publisher', 'Yahoo Finance'),
                        published_date=datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                        symbols=[symbol]
                    )
                    news_items.append(news_item)
                    
            logger.info(f"Scraped {len(news_items)} financial news items")
            return news_items
            
        except Exception as e:
            logger.error(f"Error scraping financial news: {e}")
            return []
    
    async def scrape_sec_filings(self, cik: str) -> List[Dict[str, Any]]:
        """Scrape SEC filings for a company"""
        filings = []
        
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
            headers = {
                'User-Agent': 'Credit Intelligence Platform contact@example.com'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    data = await response.json()
                    
            recent_filings = data.get('filings', {}).get('recent', {})
            forms = recent_filings.get('form', [])
            filing_dates = recent_filings.get('filingDate', [])
            accession_numbers = recent_filings.get('accessionNumber', [])
            
            for i, form in enumerate(forms[:10]):  # Last 10 filings
                if form in ['10-K', '10-Q', '8-K']:
                    filing = {
                        'form': form,
                        'filing_date': filing_dates[i],
                        'accession_number': accession_numbers[i],
                        'url': f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_numbers[i].replace('-', '')}/{accession_numbers[i]}-index.htm"
                    }
                    filings.append(filing)
                    
            logger.info(f"Found {len(filings)} SEC filings for CIK {cik}")
            return filings
            
        except Exception as e:
            logger.error(f"Error scraping SEC filings: {e}")
            return []
    
    async def scrape_earnings_calendar(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Scrape earnings calendar data"""
        earnings_data = []
        
        try:
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                calendar = ticker.calendar
                
                if calendar is not None and not calendar.empty:
                    for date, row in calendar.iterrows():
                        earnings = {
                            'symbol': symbol,
                            'earnings_date': date,
                            'eps_estimate': row.get('Earnings Estimate'),
                            'revenue_estimate': row.get('Revenue Estimate'),
                            'source': 'Yahoo Finance'
                        }
                        earnings_data.append(earnings)
                        
            logger.info(f"Scraped earnings data for {len(earnings_data)} events")
            return earnings_data
            
        except Exception as e:
            logger.error(f"Error scraping earnings calendar: {e}")
            return []
    
    async def scrape_insider_trading(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Scrape insider trading data"""
        insider_data = []
        
        try:
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                insider_trades = ticker.insider_transactions
                
                if insider_trades is not None and not insider_trades.empty:
                    for _, trade in insider_trades.head(10).iterrows():
                        insider = {
                            'symbol': symbol,
                            'insider_name': trade.get('Insider'),
                            'transaction_type': trade.get('Transaction'),
                            'shares': trade.get('Shares'),
                            'price': trade.get('Price'),
                            'date': trade.get('Date'),
                            'source': 'Yahoo Finance'
                        }
                        insider_data.append(insider)
                        
            logger.info(f"Scraped {len(insider_data)} insider trading records")
            return insider_data
            
        except Exception as e:
            logger.error(f"Error scraping insider trading: {e}")
            return []
    
    async def scrape_analyst_recommendations(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Scrape analyst recommendations"""
        recommendations = []
        
        try:
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                recs = ticker.recommendations
                
                if recs is not None and not recs.empty:
                    latest_recs = recs.tail(5)  # Last 5 recommendations
                    
                    for _, rec in latest_recs.iterrows():
                        recommendation = {
                            'symbol': symbol,
                            'firm': rec.get('Firm'),
                            'to_grade': rec.get('To Grade'),
                            'from_grade': rec.get('From Grade'),
                            'action': rec.get('Action'),
                            'date': rec.name,  # Index is the date
                            'source': 'Yahoo Finance'
                        }
                        recommendations.append(recommendation)
                        
            logger.info(f"Scraped {len(recommendations)} analyst recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error scraping analyst recommendations: {e}")
            return []
    
    async def scrape_economic_indicators(self) -> List[Dict[str, Any]]:
        """Scrape economic indicators from FRED API"""
        if 'fred' not in self.api_keys:
            logger.warning("FRED API key not found")
            return []
            
        indicators = []
        api_key = self.api_keys['fred']
        
        # Key economic indicators
        series_ids = {
            'GDP': 'GDP',
            'Unemployment Rate': 'UNRATE',
            'Inflation Rate': 'CPIAUCSL',
            'Federal Funds Rate': 'FEDFUNDS',
            'Consumer Confidence': 'UMCSENT'
        }
        
        try:
            for name, series_id in series_ids.items():
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': series_id,
                    'api_key': api_key,
                    'file_type': 'json',
                    'limit': 1,
                    'sort_order': 'desc'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        data = await response.json()
                        
                observations = data.get('observations', [])
                if observations:
                    latest = observations[0]
                    indicator = {
                        'name': name,
                        'series_id': series_id,
                        'value': float(latest['value']) if latest['value'] != '.' else None,
                        'date': latest['date'],
                        'source': 'FRED'
                    }
                    indicators.append(indicator)
                    
            logger.info(f"Scraped {len(indicators)} economic indicators")
            return indicators
            
        except Exception as e:
            logger.error(f"Error scraping economic indicators: {e}")
            return []
    
    async def scrape_all_financial_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Scrape all financial data sources"""
        results = {}
        
        # Run all scrapers concurrently
        tasks = [
            self.scrape_yahoo_finance(symbols),
            self.scrape_alpha_vantage(symbols[:5]),  # Limit due to rate limits
            self.scrape_financial_news(symbols),
            self.scrape_earnings_calendar(symbols),
            self.scrape_insider_trading(symbols),
            self.scrape_analyst_recommendations(symbols),
            self.scrape_economic_indicators()
        ]
        
        task_names = [
            'stock_data_yahoo',
            'stock_data_alpha_vantage', 
            'financial_news',
            'earnings_calendar',
            'insider_trading',
            'analyst_recommendations',
            'economic_indicators'
        ]
        
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(task_results):
            if isinstance(result, list):
                results[task_names[i]] = result
            elif isinstance(result, Exception):
                logger.error(f"Error in {task_names[i]}: {result}")
                results[task_names[i]] = []
                
        return results
    
    def get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            return sp500_table['Symbol'].tolist()
        except Exception as e:
            logger.error(f"Error getting S&P 500 symbols: {e}")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Fallback list
