import asyncio, aiohttp
import feedparser
from datetime import datetime
from typing import Dict, List, Any
import logging

from stage1_data_ingestion.specialized_scrapers import InternationalSource
from ..data_processing.data_models import DataPoint

class InternationalScraper:
    def __init__(self):
        self.sources = self._load_international_sources()
    
    def _create_international_scraper(self):
        """Create international news scraper with sources"""
        class InternationalScraper:
            def __init__(self):
                self.sources = {
                    'german': [
                        {'name': 'Handelsblatt', 'rss': 'https://www.handelsblatt.com/contentexport/feed/schlagzeilen'},
                        {'name': 'Manager Magazin', 'rss': 'https://www.manager-magazin.de/news/rss/'},
                    ],
                    'french': [
                        {'name': 'Les Echos', 'rss': 'https://www.lesechos.fr/rss/lesechos_actualites_accueil.xml'},
                        {'name': 'La Tribune', 'rss': 'https://www.latribune.fr/rss/actualites.html'},
                    ],
                    'spanish': [
                        {'name': 'Expansión', 'rss': 'https://www.expansion.com/rss/portada.xml'},
                        {'name': 'Cinco Días', 'rss': 'https://cincodias.elpais.com/rss/cincodias/portada.xml'},
                    ]
                }
                self.sources = {
                    'german': [
                        {'name': 'Handelsblatt', 'rss': 'https://www.handelsblatt.com/contentexport/feed/schlagzeilen'},
                        {'name': 'Manager Magazin', 'rss': 'https://www.manager-magazin.de/news/rss/'},
                    ],
                    'french': [
                        {'name': 'Les Echos', 'rss': 'https://www.lesechos.fr/rss/lesechos_actualites_accueil.xml'},
                        {'name': 'La Tribune', 'rss': 'https://www.latribune.fr/rss/actualites.html'},
                    ],
                    'spanish': [
                        {'name': 'Expansión', 'rss': 'https://www.expansion.com/rss/portada.xml'},
                        {'name': 'Cinco Días', 'rss': 'https://cincodias.elpais.com/rss/cincodias/portada.xml'},
                    ]
                }
            
            async def scrape_international_news(self, language: str, company_registry) -> List[DataPoint]:
                data_points = []
                if language not in self.sources:
                    return data_points
                
                for source in self.sources[language]:
                    try:
                        feed = feedparser.parse(source['rss'])
                        for entry in feed.entries[:10]:
                            content = entry.get('summary', '') + ' ' + entry.get('description', '')
                            mentioned_companies = self._extract_companies(content, company_registry)
                            
                            for company in mentioned_companies:
                                data_point = DataPoint(
                                    source_type='international_rss',
                                    source_name=source['name'],
                                    company_ticker=company,
                                    company_name=None,
                                    content_type='news',
                                    language=language,
                                    title=entry.get('title'),
                                    content=content,
                                    url=entry.get('link'),
                                    published_date=datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else None,
                                    metadata={'source_language': language, 'author': entry.get('author')}
                                )
                                data_points.append(data_point)
                    except Exception as e:
                        logger.error(f"Error scraping {source['name']}: {e}")
                
                return data_points
            
            def _extract_companies(self, text: str, company_registry) -> List[str]:
                mentioned = []
                text_upper = text.upper()
                for ticker, info in company_registry.companies.items():
                    if ticker.split('.')[0] in text_upper or info['name'].upper() in text_upper:
                        mentioned.append(ticker)
                return mentioned
        
        return InternationalScraper()
    
    def _load_international_sources(self) -> Dict[str, List[InternationalSource]]:
        return {
            'german': [
                InternationalSource('Handelsblatt', 'DE', 'de', 
                    'https://www.handelsblatt.com/contentexport/feed/schlagzeilen', 
                    'https://www.handelsblatt.com', '.o-teaser__content'),
                InternationalSource('Manager Magazin', 'DE', 'de',
                    'https://www.manager-magazin.de/news/rss/',
                    'https://www.manager-magazin.de', '.article-content'),
                InternationalSource('Börsen-Zeitung', 'DE', 'de',
                    'https://www.boersen-zeitung.de/rss/news.xml',
                    'https://www.boersen-zeitung.de', '.article-body')
            ],
            'french': [
                InternationalSource('Les Echos', 'FR', 'fr',
                    'https://www.lesechos.fr/rss/lesechos_actualites_accueil.xml',
                    'https://www.lesechos.fr', '.post-content'),
                InternationalSource('La Tribune', 'FR', 'fr',
                    'https://www.latribune.fr/rss/actualites.html',
                    'https://www.latribune.fr', '.article-body'),
                InternationalSource('Boursorama', 'FR', 'fr',
                    'https://www.boursorama.com/rss/actualites/',
                    'https://www.boursorama.com', '.c-article-body')
            ],
            'spanish': [
                InternationalSource('Expansión', 'ES', 'es',
                    'https://www.expansion.com/rss/portada.xml',
                    'https://www.expansion.com', '.article-body'),
                InternationalSource('Cinco Días', 'ES', 'es',
                    'https://cincodias.elpais.com/rss/cincodias/portada.xml',
                    'https://cincodias.elpais.com', '.articulo-cuerpo'),
                InternationalSource('El Economista', 'ES', 'es',
                    'https://www.eleconomista.es/rss/rss.php',
                    'https://www.eleconomista.es', '.articulo-desarrollo')
            ],
            'japanese': [
                InternationalSource('Nikkei', 'JP', 'ja',
                    'https://www.nikkei.com/news/category/markets.rss',
                    'https://www.nikkei.com', '.article-body'),
                InternationalSource('Bloomberg Japan', 'JP', 'ja',
                    'https://www.bloomberg.co.jp/feeds/japan-economy.rss',
                    'https://www.bloomberg.co.jp', '.body-content'),
                InternationalSource('Kabutan', 'JP', 'ja',
                    'https://kabutan.jp/news/rss',
                    'https://kabutan.jp', '.news-content')
            ],
            'chinese': [
                InternationalSource('Sina Finance', 'CN', 'zh',
                    'http://finance.sina.com.cn/roll/index.d.html?cid=56588',
                    'http://finance.sina.com.cn', '.article-content'),
                InternationalSource('163 Money', 'CN', 'zh',
                    'http://money.163.com/special/002557S6/rss_jsxw.xml',
                    'http://money.163.com', '.post-content'),
                InternationalSource('Eastmoney', 'CN', 'zh',
                    'http://stock.eastmoney.com/news/',
                    'http://stock.eastmoney.com', '.news-body')
            ],
            'hindi': [
                InternationalSource('LiveMint', 'IN', 'hi',
                    'https://www.livemint.com/rss/companies',
                    'https://www.livemint.com', '.paywall'),
                InternationalSource('Economic Times Hindi', 'IN', 'hi',
                    'https://economictimes.indiatimes.com/hindi/rss/markets.cms',
                    'https://economictimes.indiatimes.com', '.article-content'),
                InternationalSource('MoneyControl Hindi', 'IN', 'hi',
                    'https://hindi.moneycontrol.com/rss/stock-market.xml',
                    'https://hindi.moneycontrol.com', '.article-body')
            ]
        }
    
    async def scrape_international_news(self, language: str, company_registry) -> List[Dict[str, Any]]:
        """Scrape news from international sources in specified language"""
        if language not in self.sources:
            logger.warning(f"Language {language} not supported")
            return []
        
        data_points = []
        sources = self.sources[language]
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for source in sources:
                task = self._scrape_source(session, source, company_registry)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error scraping international source: {result}")
                else:
                    data_points.extend(result)
        
        return data_points
    
    async def _scrape_source(self, session: aiohttp.ClientSession, source: InternationalSource, company_registry) -> List[Dict[str, Any]]:
        """Scrape a single international source"""
        data_points = []
        
        try:
            # Try RSS first
            data_points.extend(await self._scrape_rss(source, company_registry))
            
            # If RSS fails or returns few results, try web scraping
            if len(data_points) < 5:
                web_data = await self._scrape_website(session, source, company_registry)
                data_points.extend(web_data)
        
        except Exception as e:
            logger.error(f"Error scraping {source.name}: {e}")
        
        return data_points
    
    async def _scrape_rss(self, source: InternationalSource, company_registry) -> List[Dict[str, Any]]:
        """Scrape RSS feed"""
        data_points = []
        
        try:
            feed = feedparser.parse(source.rss_url)
            
            for entry in feed.entries[:20]:  # Limit to 20 recent entries
                content = entry.get('summary', '') + ' ' + entry.get('description', '')
                mentioned_companies = self._extract_companies_multilingual(content, company_registry, source.language)
                
                for company in mentioned_companies:
                    data_points.append({
                        'source_type': 'international_rss',
                        'source_name': source.name,
                        'company_ticker': company,
                        'content_type': 'news',
                        'language': source.language,
                        'title': entry.get('title'),
                        'content': content,
                        'url': entry.get('link'),
                        'published_date': datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else None,
                        'metadata': {
                            'country': source.country,
                            'source_language': source.language,
                            'author': entry.get('author')
                        }
                    })
        
        except Exception as e:
            logger.error(f"Error scraping RSS for {source.name}: {e}")
        
        return data_points
    
    async def _scrape_website(self, session: aiohttp.ClientSession, source: InternationalSource, company_registry) -> List[Dict[str, Any]]:
        """Scrape website directly"""
        data_points = []
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept-Language': f'{source.language},en;q=0.9'
            }
            
            async with session.get(source.base_url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find articles using the selector
                    articles = soup.select(source.selector)
                    
                    for article in articles[:10]:  # Limit to 10 articles
                        title = article.find(['h1', 'h2', 'h3', 'h4'])
                        title_text = title.get_text(strip=True) if title else ''
                        content = article.get_text(strip=True)
                        
                        mentioned_companies = self._extract_companies_multilingual(content, company_registry, source.language)
                        
                        for company in mentioned_companies:
                            data_points.append({
                                'source_type': 'international_web',
                                'source_name': source.name,
                                'company_ticker': company,
                                'content_type': 'news',
                                'language': source.language,
                                'title': title_text,
                                'content': content[:2000],  # Limit content length
                                'url': source.base_url,
                                'published_date': datetime.utcnow(),
                                'metadata': {
                                    'country': source.country,
                                    'source_language': source.language,
                                    'scraped_from': 'website'
                                }
                            })
        
        except Exception as e:
            logger.error(f"Error web scraping {source.name}: {e}")
        
        return data_points
    
    def _extract_companies_multilingual(self, text: str, company_registry, language: str) -> List[str]:
        """Extract company mentions with multilingual support"""
        mentioned_companies = []
        text_upper = text.upper()
        
        # Define company name translations/variations
        company_variations = {
            'de': {
                'APPLE': ['APPLE', 'APFEL'],
                'MICROSOFT': ['MICROSOFT', 'MIKRO'],
                'AMAZON': ['AMAZON', 'AMAZONE'],
                'GOOGLE': ['GOOGLE', 'ALPHABET']
            },
            'fr': {
                'APPLE': ['APPLE', 'POMME'],
                'MICROSOFT': ['MICROSOFT'],
                'AMAZON': ['AMAZON', 'AMAZONE'],
                'GOOGLE': ['GOOGLE', 'ALPHABET']
            },
            'es': {
                'APPLE': ['APPLE', 'MANZANA'],
                'MICROSOFT': ['MICROSOFT'],
                'AMAZON': ['AMAZON'],
                'GOOGLE': ['GOOGLE', 'ALPHABET']
            },
            'ja': {
                'APPLE': ['APPLE', 'アップル'],
                'MICROSOFT': ['MICROSOFT', 'マイクロソフト'],
                'AMAZON': ['AMAZON', 'アマゾン'],
                'GOOGLE': ['GOOGLE', 'グーグル']
            },
            'zh': {
                'APPLE': ['APPLE', '苹果'],
                'MICROSOFT': ['MICROSOFT', '微软'],
                'AMAZON': ['AMAZON', '亚马逊'],
                'GOOGLE': ['GOOGLE', '谷歌']
            },
            'hi': {
                'APPLE': ['APPLE', 'एप्पल'],
                'MICROSOFT': ['MICROSOFT', 'माइक्रोसॉफ्ट'],
                'AMAZON': ['AMAZON', 'अमेज़न'],
                'GOOGLE': ['GOOGLE', 'गूगल']
            }
        }
        
        # Check for ticker symbols and company names
        for ticker, info in company_registry.companies.items():
            # Check ticker
            if ticker.split('.')[0] in text_upper:
                mentioned_companies.append(ticker)
                continue
            
            # Check original company name
            if info['name'].upper() in text_upper:
                mentioned_companies.append(ticker)
                continue
            
            # Check language-specific variations
            if language in company_variations:
                company_key = info['name'].upper().split()[0]  # Get first word
                if company_key in company_variations[language]:
                    for variation in company_variations[language][company_key]:
                        if variation in text_upper:
                            mentioned_companies.append(ticker)
                            break
        
        return list(set(mentioned_companies))  # Remove duplicates