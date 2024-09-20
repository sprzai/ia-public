import requests
from bs4 import BeautifulSoup
import pandas as pd
from src.utils.logging_config import setup_logging

logger = setup_logging()

class Scraper:
    def __init__(self):
        self.session = requests.Session()

    def scrape_page(self, url):
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            properties = []
            items = soup.find_all('li', class_='ui-search-layout__item')
            
            for item in items:
                property_data = self._extract_property_data(item)
                if property_data:
                    properties.append(property_data)
            
            logger.info(f"Extraídas {len(properties)} propiedades de {url}")
            return properties
        except requests.RequestException as e:
            logger.error(f"Error al hacer scraping de {url}: {str(e)}")
            raise

    def _extract_property_data(self, item):
        try:
            price_elem = item.find('span', class_='andes-money-amount__fraction')
            currency_elem = item.find('span', class_='andes-money-amount__currency-symbol')
            location_elem = item.find('span', class_='ui-search-item__location')
            attributes = item.find_all('li', class_='ui-search-card-attributes__attribute')
            
            if price_elem and currency_elem and location_elem and len(attributes) >= 3:
                return {
                    'price': price_elem.text.strip(),
                    'currency': currency_elem.text.strip(),
                    'location': location_elem.text.strip(),
                    'bedrooms': attributes[0].text.strip(),
                    'bathrooms': attributes[1].text.strip(),
                    'area': attributes[2].text.strip()
                }
        except AttributeError as e:
            logger.warning(f"Error al extraer datos de una propiedad: {str(e)}")
        return None

    def save_to_csv(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Datos guardados en {filename}")