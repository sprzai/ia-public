import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
BASE_URL = "https://www.portalinmobiliario.com/venta/casa/"
REGIONS = ["Valdivia", "Santiago", "Concepcion"]
MAX_RETRIES = 3
PAGES_PER_REGION = 5