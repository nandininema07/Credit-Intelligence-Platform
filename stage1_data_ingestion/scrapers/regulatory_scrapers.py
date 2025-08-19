"""
Regulatory data scrapers for SEC, FINRA, and other regulatory sources.
Handles compliance and regulatory filing data collection.
"""

import requests
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import json

logger = logging.getLogger(__name__)

@dataclass
class RegulatoryFiling:
    """Data class for regulatory filings"""
    company_name: str
    cik: str
    form_type: str
    filing_date: datetime
    accession_number: str
    file_url: str
    description: str
    source: str
    file_size: Optional[int] = None
    items: Optional[List[str]] = None

@dataclass
class EnforcementAction:
    """Data class for regulatory enforcement actions"""
    respondent: str
    action_type: str
    date: datetime
    description: str
    penalty_amount: Optional[float]
    source: str
    url: str
    status: str

class RegulatoryScrapers:
    """Regulatory data scrapers for compliance monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_keys = config.get('api_keys', {})
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Credit Intelligence Platform contact@example.com'
        })
        
    async def scrape_sec_filings(self, cik: str, forms: List[str] = None) -> List[RegulatoryFiling]:
        """Scrape SEC filings for a company"""
        if forms is None:
            forms = ['10-K', '10-Q', '8-K', 'DEF 14A', '13F-HR']
            
        filings = []
        
        try:
            # Get company submissions
            url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.session.headers) as response:
                    data = await response.json()
                    
            company_name = data.get('name', 'Unknown')
            recent_filings = data.get('filings', {}).get('recent', {})
            
            form_list = recent_filings.get('form', [])
            filing_dates = recent_filings.get('filingDate', [])
            accession_numbers = recent_filings.get('accessionNumber', [])
            primary_docs = recent_filings.get('primaryDocument', [])
            
            for i, form in enumerate(form_list):
                if form in forms and i < len(filing_dates):
                    filing = RegulatoryFiling(
                        company_name=company_name,
                        cik=cik,
                        form_type=form,
                        filing_date=datetime.strptime(filing_dates[i], '%Y-%m-%d'),
                        accession_number=accession_numbers[i],
                        file_url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_numbers[i].replace('-', '')}/{primary_docs[i]}",
                        description=f"{form} filing for {company_name}",
                        source='SEC EDGAR'
                    )
                    filings.append(filing)
                    
            logger.info(f"Scraped {len(filings)} SEC filings for CIK {cik}")
            return filings
            
        except Exception as e:
            logger.error(f"Error scraping SEC filings: {e}")
            return []
    
    async def scrape_finra_actions(self, firm_name: str = None) -> List[EnforcementAction]:
        """Scrape FINRA enforcement actions"""
        actions = []
        
        try:
            # FINRA disciplinary actions search
            url = "https://www.finra.org/rules-guidance/oversight-enforcement/finra-disciplinary-actions"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    html = await response.text()
                    
            soup = BeautifulSoup(html, 'html.parser')
            
            # Parse enforcement actions (simplified)
            for action_elem in soup.find_all('div', class_='enforcement-action')[:20]:
                title_elem = action_elem.find('h3') or action_elem.find('h2')
                if not title_elem:
                    continue
                    
                date_elem = action_elem.find('span', class_='date')
                description_elem = action_elem.find('p')
                
                action = EnforcementAction(
                    respondent=title_elem.get_text(strip=True),
                    action_type='Disciplinary Action',
                    date=datetime.now(),  # Would need to parse actual date
                    description=description_elem.get_text(strip=True) if description_elem else '',
                    penalty_amount=None,
                    source='FINRA',
                    url=url,
                    status='Published'
                )
                actions.append(action)
                
            logger.info(f"Scraped {len(actions)} FINRA actions")
            return actions
            
        except Exception as e:
            logger.error(f"Error scraping FINRA actions: {e}")
            return []
    
    async def scrape_cftc_actions(self) -> List[EnforcementAction]:
        """Scrape CFTC enforcement actions"""
        actions = []
        
        try:
            url = "https://www.cftc.gov/LawRegulation/Enforcement/EnforcementActions"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    html = await response.text()
                    
            soup = BeautifulSoup(html, 'html.parser')
            
            # Parse CFTC actions (simplified)
            for action_elem in soup.find_all('div', class_='enforcement-item')[:15]:
                title_elem = action_elem.find('a')
                if not title_elem:
                    continue
                    
                action = EnforcementAction(
                    respondent=title_elem.get_text(strip=True),
                    action_type='CFTC Enforcement',
                    date=datetime.now(),
                    description='',
                    penalty_amount=None,
                    source='CFTC',
                    url=f"https://www.cftc.gov{title_elem.get('href', '')}",
                    status='Published'
                )
                actions.append(action)
                
            logger.info(f"Scraped {len(actions)} CFTC actions")
            return actions
            
        except Exception as e:
            logger.error(f"Error scraping CFTC actions: {e}")
            return []
    
    async def scrape_fed_announcements(self) -> List[Dict[str, Any]]:
        """Scrape Federal Reserve announcements"""
        announcements = []
        
        try:
            url = "https://www.federalreserve.gov/feeds/press_all.xml"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    xml_content = await response.text()
                    
            root = ET.fromstring(xml_content)
            
            for item in root.findall('.//item')[:10]:
                title = item.find('title').text if item.find('title') is not None else ''
                link = item.find('link').text if item.find('link') is not None else ''
                pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ''
                description = item.find('description').text if item.find('description') is not None else ''
                
                announcement = {
                    'title': title,
                    'url': link,
                    'published_date': pub_date,
                    'description': description,
                    'source': 'Federal Reserve',
                    'type': 'Press Release'
                }
                announcements.append(announcement)
                
            logger.info(f"Scraped {len(announcements)} Fed announcements")
            return announcements
            
        except Exception as e:
            logger.error(f"Error scraping Fed announcements: {e}")
            return []
    
    async def scrape_occ_actions(self) -> List[EnforcementAction]:
        """Scrape OCC enforcement actions"""
        actions = []
        
        try:
            url = "https://www.occ.gov/news-issuances/news-releases/index-enforcement-actions.html"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    html = await response.text()
                    
            soup = BeautifulSoup(html, 'html.parser')
            
            # Parse OCC actions
            for action_elem in soup.find_all('div', class_='news-item')[:10]:
                title_elem = action_elem.find('h3') or action_elem.find('h2')
                if not title_elem:
                    continue
                    
                link_elem = title_elem.find('a')
                date_elem = action_elem.find('span', class_='date')
                
                action = EnforcementAction(
                    respondent=title_elem.get_text(strip=True),
                    action_type='OCC Enforcement',
                    date=datetime.now(),  # Would parse actual date
                    description='',
                    penalty_amount=None,
                    source='OCC',
                    url=f"https://www.occ.gov{link_elem.get('href', '')}" if link_elem else '',
                    status='Published'
                )
                actions.append(action)
                
            logger.info(f"Scraped {len(actions)} OCC actions")
            return actions
            
        except Exception as e:
            logger.error(f"Error scraping OCC actions: {e}")
            return []
    
    async def scrape_all_regulatory_data(self, ciks: List[str]) -> Dict[str, Any]:
        """Scrape all regulatory data sources"""
        results = {}
        
        # Scrape SEC filings for each CIK
        sec_tasks = [self.scrape_sec_filings(cik) for cik in ciks]
        
        # Other regulatory sources
        other_tasks = [
            self.scrape_finra_actions(),
            self.scrape_cftc_actions(),
            self.scrape_fed_announcements(),
            self.scrape_occ_actions()
        ]
        
        # Execute all tasks
        all_tasks = sec_tasks + other_tasks
        task_results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Process SEC results
        results['sec_filings'] = []
        for i, result in enumerate(task_results[:len(ciks)]):
            if isinstance(result, list):
                results['sec_filings'].extend(result)
                
        # Process other regulatory results
        task_names = ['finra_actions', 'cftc_actions', 'fed_announcements', 'occ_actions']
        for i, result in enumerate(task_results[len(ciks):]):
            if isinstance(result, list):
                results[task_names[i]] = result
            elif isinstance(result, Exception):
                logger.error(f"Error in {task_names[i]}: {result}")
                results[task_names[i]] = []
                
        return results
