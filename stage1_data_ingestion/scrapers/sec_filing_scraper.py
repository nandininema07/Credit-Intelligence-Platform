"""
SEC Filing scraper for regulatory filings data.
Scrapes 10-K, 10-Q, 8-K filings and other regulatory documents.
"""

import asyncio
import logging
import json
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
import time

logger = logging.getLogger(__name__)

@dataclass
class SECFiling:
    """SEC Filing data structure"""
    company_ticker: str
    company_name: str
    cik: str
    filing_type: str
    filing_date: datetime
    acceptance_datetime: datetime
    accession_number: str
    document_url: str
    html_url: str
    filing_size: Optional[int]
    description: str
    content: str
    metadata: Dict[str, Any]

class SECFilingScraper:
    """SEC EDGAR database scraper for regulatory filings"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = "https://www.sec.gov"
        self.edgar_api_url = "https://data.sec.gov"
        self.session_headers = {
            'User-Agent': 'Credit Intelligence Platform contact@credtech.com',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        
        # Rate limiting - SEC requires 10 requests per second max
        self.rate_limit_delay = 0.1  # 100ms between requests
        
        # Filing types to track
        self.filing_types = [
            '10-K',    # Annual report
            '10-Q',    # Quarterly report
            '8-K',     # Current report
            '10-K/A',  # Annual report amendment
            '10-Q/A',  # Quarterly report amendment
            '8-K/A',   # Current report amendment
            'DEF 14A', # Proxy statement
            'S-1',     # Registration statement
            '20-F',    # Annual report (foreign companies)
            '6-K'      # Report of foreign private issuer
        ]
    
    async def get_company_cik(self, ticker: str) -> Optional[str]:
        """Get CIK (Central Index Key) for a company ticker"""
        try:
            async with aiohttp.ClientSession(headers=self.session_headers) as session:
                # Use SEC company tickers JSON
                url = f"{self.edgar_api_url}/files/company_tickers.json"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Search for ticker in the data
                        for entry in data.values():
                            if entry.get('ticker', '').upper() == ticker.upper():
                                cik = str(entry.get('cik_str', '')).zfill(10)
                                logger.info(f"Found CIK {cik} for ticker {ticker}")
                                return cik
                        
                        logger.warning(f"CIK not found for ticker {ticker}")
                        return None
                    else:
                        logger.error(f"Failed to fetch company tickers: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error getting CIK for {ticker}: {e}")
            return None
    
    async def get_recent_filings(self, cik: str, filing_types: List[str] = None, 
                               limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent filings for a company by CIK"""
        if not filing_types:
            filing_types = self.filing_types
            
        try:
            async with aiohttp.ClientSession(headers=self.session_headers) as session:
                # Use SEC submissions API
                url = f"{self.edgar_api_url}/submissions/CIK{cik}.json"
                
                await asyncio.sleep(self.rate_limit_delay)
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        filings = []
                        recent_filings = data.get('filings', {}).get('recent', {})
                        
                        if not recent_filings:
                            return filings
                        
                        # Extract filing data
                        forms = recent_filings.get('form', [])
                        filing_dates = recent_filings.get('filingDate', [])
                        accession_numbers = recent_filings.get('accessionNumber', [])
                        primary_documents = recent_filings.get('primaryDocument', [])
                        
                        for i in range(min(len(forms), limit)):
                            form_type = forms[i]
                            
                            if form_type in filing_types:
                                filing = {
                                    'form_type': form_type,
                                    'filing_date': filing_dates[i] if i < len(filing_dates) else None,
                                    'accession_number': accession_numbers[i] if i < len(accession_numbers) else None,
                                    'primary_document': primary_documents[i] if i < len(primary_documents) else None,
                                    'cik': cik
                                }
                                filings.append(filing)
                        
                        return filings
                    else:
                        logger.error(f"Failed to fetch filings for CIK {cik}: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error getting filings for CIK {cik}: {e}")
            return []
    
    async def get_filing_content(self, cik: str, accession_number: str, 
                               primary_document: str) -> Optional[str]:
        """Get the content of a specific filing"""
        try:
            # Format accession number for URL (remove dashes)
            accession_no_dashes = accession_number.replace('-', '')
            
            # Construct document URL
            doc_url = f"{self.base_url}/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_document}"
            
            async with aiohttp.ClientSession(headers=self.session_headers) as session:
                await asyncio.sleep(self.rate_limit_delay)
                
                async with session.get(doc_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse HTML/XML content
                        if content.strip().startswith('<'):
                            soup = BeautifulSoup(content, 'html.parser')
                            
                            # Extract text content, removing excessive whitespace
                            text_content = soup.get_text()
                            text_content = re.sub(r'\s+', ' ', text_content).strip()
                            
                            return text_content[:50000]  # Limit content size
                        else:
                            return content[:50000]
                    else:
                        logger.warning(f"Failed to fetch filing content: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error getting filing content: {e}")
            return None
    
    async def scrape_company_filings(self, ticker: str, days_back: int = 30) -> List[SECFiling]:
        """Scrape recent SEC filings for a company"""
        filings = []
        
        try:
            # Get company CIK
            cik = await self.get_company_cik(ticker)
            if not cik:
                return filings
            
            # Get recent filings
            recent_filings = await self.get_recent_filings(cik, limit=20)
            
            # Filter by date if specified
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for filing_data in recent_filings:
                try:
                    filing_date_str = filing_data.get('filing_date')
                    if not filing_date_str:
                        continue
                    
                    filing_date = datetime.strptime(filing_date_str, '%Y-%m-%d')
                    
                    # Skip if too old
                    if filing_date < cutoff_date:
                        continue
                    
                    # Get filing content
                    content = ""
                    primary_doc = filing_data.get('primary_document')
                    accession_number = filing_data.get('accession_number')
                    
                    if primary_doc and accession_number:
                        content = await self.get_filing_content(cik, accession_number, primary_doc)
                        if not content:
                            content = f"Filing {filing_data.get('form_type')} submitted on {filing_date_str}"
                    
                    # Create SEC filing object
                    sec_filing = SECFiling(
                        company_ticker=ticker,
                        company_name="",  # Will be filled from company registry
                        cik=cik,
                        filing_type=filing_data.get('form_type', ''),
                        filing_date=filing_date,
                        acceptance_datetime=filing_date,  # Simplified
                        accession_number=accession_number or '',
                        document_url=f"{self.base_url}/Archives/edgar/data/{cik}/{accession_number.replace('-', '')}/{primary_doc}" if accession_number and primary_doc else '',
                        html_url=f"{self.base_url}/cgi-bin/viewer?action=view&cik={cik}&accession_number={accession_number}" if accession_number else '',
                        filing_size=len(content) if content else 0,
                        description=f"{filing_data.get('form_type')} filing for {ticker}",
                        content=content or '',
                        metadata={
                            'cik': cik,
                            'accession_number': accession_number,
                            'primary_document': primary_doc,
                            'form_type': filing_data.get('form_type')
                        }
                    )
                    
                    filings.append(sec_filing)
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"Error processing filing for {ticker}: {e}")
                    continue
            
            logger.info(f"Scraped {len(filings)} SEC filings for {ticker}")
            return filings
            
        except Exception as e:
            logger.error(f"Error scraping SEC filings for {ticker}: {e}")
            return []
    
    async def scrape_multiple_companies(self, tickers: List[str]) -> List[SECFiling]:
        """Scrape SEC filings for multiple companies"""
        all_filings = []
        
        for ticker in tickers:
            try:
                company_filings = await self.scrape_company_filings(ticker)
                all_filings.extend(company_filings)
                
                # Add delay between companies to respect rate limits
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error scraping filings for {ticker}: {e}")
                continue
        
        return all_filings
    
    def get_filing_summary(self, filings: List[SECFiling]) -> Dict[str, Any]:
        """Get summary statistics of scraped filings"""
        summary = {
            'total_filings': len(filings),
            'by_type': {},
            'by_company': {},
            'date_range': {'earliest': None, 'latest': None},
            'total_content_size': 0
        }
        
        for filing in filings:
            # By type
            filing_type = filing.filing_type
            summary['by_type'][filing_type] = summary['by_type'].get(filing_type, 0) + 1
            
            # By company
            ticker = filing.company_ticker
            summary['by_company'][ticker] = summary['by_company'].get(ticker, 0) + 1
            
            # Date range
            if not summary['date_range']['earliest'] or filing.filing_date < summary['date_range']['earliest']:
                summary['date_range']['earliest'] = filing.filing_date
            if not summary['date_range']['latest'] or filing.filing_date > summary['date_range']['latest']:
                summary['date_range']['latest'] = filing.filing_date
            
            # Content size
            summary['total_content_size'] += len(filing.content)
        
        return summary

# Integration with the multi-source collector
class SECDataCollector:
    """SEC data collector for integration with multi-source pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.scraper = SECFilingScraper(config)
    
    async def collect_sec_data(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """Collect SEC filing data in standardized format"""
        from .multi_source_collector import DataPoint, DataProcessor
        
        data_points = []
        
        # Limit to top companies to respect rate limits
        limited_tickers = tickers[:10]
        
        filings = await self.scraper.scrape_multiple_companies(limited_tickers)
        
        for filing in filings:
            data_point = DataPoint(
                source_type='sec_filing',
                source_name='sec_edgar',
                company_ticker=filing.company_ticker,
                company_name=filing.company_name,
                content_type='regulatory_filing',
                language='en',
                title=f"{filing.filing_type} Filing - {filing.company_ticker}",
                content=filing.content,
                url=filing.html_url,
                published_date=filing.filing_date,
                sentiment_score=None,  # SEC filings are neutral
                metadata={
                    'filing_type': filing.filing_type,
                    'cik': filing.cik,
                    'accession_number': filing.accession_number,
                    'filing_size': filing.filing_size,
                    'document_url': filing.document_url
                }
            )
            data_points.append(data_point)
        
        logger.info(f"Collected {len(data_points)} SEC filing data points")
        return data_points
