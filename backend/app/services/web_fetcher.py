"""Web fetching service for extracting content from URLs.

Provides functionality to fetch web pages, parse content,
and extract structured data for node creation.
"""

import logging
from typing import Optional, Dict, Any
import aiohttp
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)


class WebFetchError(Exception):
    """Raised when web fetching fails."""
    pass


class WebFetcher:
    """Fetch and parse web pages for content extraction."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    "User-Agent": "ThoughtLab Research Bot/1.0 (Research knowledge graph ingestion)"
                }
            )
        return self.session
    
    async def fetch(self, url: str) -> Dict[str, Any]:
        """Fetch a URL and extract structured content.
        
        Args:
            url: The URL to fetch
            
        Returns:
            Dict with keys: url, title, content, summary, error (if any)
            
        Raises:
            WebFetchError: If fetching or parsing fails
        """
        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status != 200:
                    raise WebFetchError(f"HTTP {response.status}: {response.reason}")
                
                content_type = response.headers.get('Content-Type', '')
                
                # Only parse HTML content
                if 'text/html' not in content_type:
                    text = await response.text()
                    return {
                        "url": url,
                        "title": url.split('/')[-1] or url,
                        "content": text[:5000],
                        "summary": f"Non-HTML content ({content_type})",
                        "success": True,
                    }
                
                html = await response.text()
                parsed = self._parse_html(html, url)
                parsed["success"] = True
                return parsed
                
        except aiohttp.ClientError as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise WebFetchError(f"Connection error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            raise WebFetchError(f"Error: {e}")
    
    def _parse_html(self, html: str, url: str) -> Dict[str, Any]:
        """Parse HTML and extract content.
        
        Args:
            html: Raw HTML content
            url: Source URL for context
            
        Returns:
            Dict with extracted title, content, summary
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title = self._extract_title(soup)
        
        # Extract main content
        content = self._extract_content(soup)
        
        # Generate summary (first few sentences)
        summary = self._generate_summary(content)
        
        return {
            "url": url,
            "title": title,
            "content": content,
            "summary": summary,
        }
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title from HTML."""
        # Try title tag
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        
        # Try h1
        h1 = soup.find('h1')
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)
        
        # Try og:title meta
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content'].strip()
        
        return "Untitled"
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML."""
        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
            script.decompose()
        
        # Try to find main content area
        main_content = None
        
        # Look for article tag
        article = soup.find('article')
        if article:
            main_content = article
        
        # Look for main tag
        if not main_content:
            main_tag = soup.find('main')
            if main_tag:
                main_content = main_tag
        
        # Look for content divs by common IDs/classes
        if not main_content:
            for selector in ['#content', '.content', '#main-content', '.main-content', 
                           '.post-content', '.article-content']:
                elem = soup.select_one(selector)
                if elem:
                    main_content = elem
                    break
        
        # Fall back to body
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            return ""
        
        # Get text and clean it up
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Limit length
        return text[:10000]  # Max 10k chars for now
    
    def _generate_summary(self, content: str, max_sentences: int = 3) -> str:
        """Generate a summary by extracting first few sentences.
        
        Args:
            content: Full content text
            max_sentences: Number of sentences to include
            
        Returns:
            Summary text
        """
        if not content:
            return ""
        
        # Split into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        # Take first N non-empty sentences
        summary_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) > 20:  # Skip very short fragments
                summary_sentences.append(sent)
            if len(summary_sentences) >= max_sentences:
                break
        
        return ' '.join(summary_sentences) if summary_sentences else content[:200]
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()


# Singleton instance
_web_fetcher: Optional[WebFetcher] = None


def get_web_fetcher() -> WebFetcher:
    """Get singleton web fetcher instance."""
    global _web_fetcher
    if _web_fetcher is None:
        _web_fetcher = WebFetcher()
    return _web_fetcher


async def fetch_url(url: str) -> Dict[str, Any]:
    """Convenience function to fetch a URL.
    
    Args:
        url: URL to fetch
        
    Returns:
        Dict with extracted data or error info
    """
    fetcher = get_web_fetcher()
    try:
        return await fetcher.fetch(url)
    except WebFetchError as e:
        return {
            "url": url,
            "error": str(e),
            "success": False,
        }
