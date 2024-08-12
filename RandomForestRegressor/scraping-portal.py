import requests
from bs4 import BeautifulSoup
import csv
import time
import random
import re

def extract_number(text):
    return re.search(r'\d+', text).group() if re.search(r'\d+', text) else ''

def scrape_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    properties = []
    
    items = soup.find_all('li', class_='ui-search-layout__item')
    
    for item in items:
        currency_element = item.find('span', class_='andes-money-amount__currency-symbol')
        price_element = item.find('span', class_='andes-money-amount__fraction')
        attributes = item.find_all('li', class_='ui-search-card-attributes__attribute')
        
        if currency_element and price_element and len(attributes) >= 3:
            currency = 'UF' if currency_element.text.strip() == 'UF' else 'Pesos'
            price = price_element.text.strip()
            bedrooms = extract_number(attributes[0].text)
            bathrooms = extract_number(attributes[1].text)
            area = extract_number(attributes[2].text)
            
            properties.append([currency, price, bedrooms, bathrooms, area])
    
    return properties

def save_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Moneda', 'Precio', 'Dormitorios', 'Baños', 'Metros Cuadrados'])
        writer.writerows(data)

base_url = "https://www.portalinmobiliario.com/venta/casa/valdivia-de-los-rios/"

all_data = []

# url = f"{base_url}_OrderId_PRICE_NoIndex_True"
# print(f"Scraping página [{url}] ...")

# page_data = scrape_page(url)
# all_data.extend(page_data)

# print(f"  Leídas {len(page_data)} propiedades de la página {1}")
# print(f"  Guardadas {len(page_data)} propiedades de la página {1}")

# pause_time = random.uniform(2,3)
# time.sleep(pause_time)

for page in range(11, 14):  # Scrape de las primeras 5 páginas
    url = f"{base_url}_Desde_{(page)*49}_OrderId_PRICE_NoIndex_True"
    print(f"Scraping página {url}...")
    
    page_data = scrape_page(url)
    all_data.extend(page_data)
    
    print(f"  Leídas {len(page_data)} propiedades de la página {page}")
    print(f"  Guardadas {len(page_data)} propiedades de la página {page}")
    
    pause_time = random.uniform(2,3)
    time.sleep(pause_time)

save_to_csv(all_data, 'propiedades_valdivia-3.csv')

print(f"Scraping completado. Se han extraído un total de {len(all_data)} propiedades.")
print("Los datos se han guardado en 'propiedades_valdivia.csv'")