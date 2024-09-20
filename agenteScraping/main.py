from src.agent.scraping_agent import ScrapingAgent
from src.utils.logging_config import setup_logging
import traceback

logger = setup_logging()

def main():
    logger.info("Iniciando el proceso de scraping")
    agent = ScrapingAgent()
    try:
        agent.run()
    except Exception as e:
        logger.error(f"Error inesperado durante la ejecución: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Proceso de scraping finalizado")

if __name__ == "__main__":
    main()