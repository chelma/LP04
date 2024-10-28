import logging
import requests
from typing import List, Dict

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ScrapingError(Exception):
    pass

def extract_text_from_page(url: str) -> Dict:
    try:
        # Fetch the web page content
        response = requests.get(url)
        response.raise_for_status()

        # Parse the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        page_structure = {}
        current_heading = None

        # Extract headings, paragraphs, lists, code blocks, and tables
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'pre', 'code', 'table']):
            tag_name = element.name
            text_content = element.get_text(separator=' ', strip=True)

            if tag_name.startswith('h'):
                # Treat headings as keys in the structure
                current_heading = text_content
                page_structure[current_heading] = []
            elif tag_name in ['p', 'pre', 'code']:
                # Add text or code under the current heading, avoiding duplication
                if current_heading:
                    if (tag_name == 'code' or tag_name == 'pre') and any(text_content == entry.get('pre', '') or text_content == entry.get('code', '') for entry in page_structure[current_heading]):
                        continue  # Skip duplicate preformatted or code blocks
                    page_structure[current_heading].append({tag_name: text_content})
            elif tag_name in ['ul', 'ol']:
                # Handle lists, avoiding duplication
                list_items = [li.get_text(separator=' ', strip=True) for li in element.find_all('li')]
                if current_heading:
                    if any(list_items == entry.get(tag_name, []) for entry in page_structure[current_heading]):
                        continue  # Skip duplicate lists
                    page_structure[current_heading].append({tag_name: list_items})
            elif tag_name == 'table':
                # Handle tables
                table_data = []
                headers = [th.get_text(separator=' ', strip=True) for th in element.find_all('th')]
                rows = element.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    row_data = [cell.get_text(separator=' ', strip=True) for cell in cells]
                    table_data.append(row_data)
                if current_heading:
                    page_structure[current_heading].append({'table': {'headers': headers, 'rows': table_data}})
        
        return page_structure
    
    except requests.exceptions.RequestException as e:
        raise ScrapingError(f"Error occurred while trying to fetch {url}: {e}")

def extract_structured_text_from_pages(url_list: List[str]) -> List[Dict]:
    text_from_pages = []

    for url in url_list:
        try:
            text_from_pages.append(extract_text_from_page(url))
        except ScrapingError as e:
            logger.error(e)
            text_from_pages.append({})

    return [extract_text_from_page(url) for url in url_list]