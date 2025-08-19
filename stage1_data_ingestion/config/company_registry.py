import yfinance as yf
from typing import Dict, List, Optional
import logging

class CompanyRegistry:
    def __init__(self):
        self.companies = self._load_companies()
    
    def _load_companies(self) -> Dict[str, Dict]:
        # Load from multiple exchanges
        exchanges = {
            'US': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V'],
            'UK': ['SHEL.L', 'AZN.L', 'ULVR.L', 'HSBA.L', 'BP.L'],
            'IN': ['TCS.NS', 'RELIANCE.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS'],
            'CN': ['0700.HK', '0939.HK', '0005.HK', '3690.HK'],
            'JP': ['7203.T', '6758.T', '9984.T', '8306.T']
        }
        
        companies = {}
        for exchange, tickers in exchanges.items():
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    companies[ticker] = {
                        'name': info.get('longName', ticker),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'country': info.get('country', exchange),
                        'website': info.get('website', ''),
                        'exchange': exchange
                    }
                except:
                    companies[ticker] = {
                        'name': ticker,
                        'sector': 'Unknown',
                        'industry': 'Unknown',
                        'country': exchange,
                        'website': '',
                        'exchange': exchange
                    }
        
        return companies
    
    def get_all_tickers(self) -> List[str]:
        return list(self.companies.keys())
    
    def get_company_info(self, ticker: str) -> Optional[Dict]:
        return self.companies.get(ticker)

