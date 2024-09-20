from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain.schema.messages import SystemMessage
from langchain.memory import ConversationBufferMemory
from src.tools.scraper import Scraper
from src.config import ANTHROPIC_API_KEY, BASE_URL, REGIONS, MAX_RETRIES, PAGES_PER_REGION
from src.utils.logging_config import setup_logging
import time
import json
import traceback

logger = setup_logging()

class ScrapingAgent:
    def __init__(self):
        self.scraper = Scraper()
        self.llm = ChatAnthropic(model="claude-3-opus-20240229", anthropic_api_key=ANTHROPIC_API_KEY)
        self.tools = [self.scrape_region]
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent = self._create_agent()
        self.performance_metrics = {region: {'success_rate': 0, 'average_time': 0} for region in REGIONS}

    def _create_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Eres un agente de scraping especializado en extraer información de propiedades inmobiliarias. Tu tarea es coordinar el proceso de scraping para diferentes regiones de Chile, manejar errores y asegurar que se obtengan todos los datos necesarios. Aprende de tus experiencias previas para mejorar tu rendimiento."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    @tool
    def scrape_region(self, region: str):
        """Scrape property data for a specific region in Chile"""
        logger.info(f"Iniciando scraping para la región: {region}")
        all_properties = []
        successful_pages = 0
        start_time = time.time()
        
        for page in range(1, PAGES_PER_REGION + 1):
            url = f"{BASE_URL}{region}/_Desde_{(page-1)*49}_OrderId_PRICE_NoIndex_True"
            
            for attempt in range(MAX_RETRIES):
                try:
                    properties = self.scraper.scrape_page(url)
                    all_properties.extend(properties)
                    successful_pages += 1
                    logger.info(f"Página {page} de {region} extraída exitosamente. Total propiedades: {len(properties)}")
                    break
                except Exception as e:
                    logger.warning(f"Intento {attempt + 1} fallido para {url}: {str(e)}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    if attempt == MAX_RETRIES - 1:
                        logger.error(f"Fallaron todos los intentos para {url}")
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        if all_properties:
            filename = f"data/raw/{region}_properties.csv"
            self.scraper.save_to_csv(all_properties, filename)
        
        end_time = time.time()
        execution_time = end_time - start_time
        success_rate = successful_pages / PAGES_PER_REGION
        
        self.update_performance_metrics(region, success_rate, execution_time)
        
        return f"Scraping completado para {region}. Se extrajeron {len(all_properties)} propiedades. Tasa de éxito: {success_rate:.2f}, Tiempo de ejecución: {execution_time:.2f} segundos."

    def update_performance_metrics(self, region, success_rate, execution_time):
        current_metrics = self.performance_metrics[region]
        current_metrics['success_rate'] = (current_metrics['success_rate'] + success_rate) / 2
        current_metrics['average_time'] = (current_metrics['average_time'] + execution_time) / 2
        logger.info(f"Métricas actualizadas para {region}: {json.dumps(current_metrics)}")

    def run(self):
        for region in REGIONS:
            try:
                result = self.agent.invoke({"input": f"Realiza scraping de propiedades en la región de {region}"})
                logger.info(f"Resultado para {region}: {result['output']}")
                self.analyze_and_improve(region)
            except Exception as e:
                logger.error(f"Error al procesar la región {region}: {str(e)}")
                logger.debug(f"Traceback completo: {traceback.format_exc()}")

        logger.info("Proceso de scraping completado para todas las regiones.")
        self.log_overall_performance()

    def analyze_and_improve(self, region):
        metrics = self.performance_metrics[region]
        if metrics['success_rate'] < 0.8:
            logger.warning(f"Baja tasa de éxito para {region}. Considerando aumentar MAX_RETRIES.")
            # Aquí podrías implementar lógica para ajustar MAX_RETRIES dinámicamente
        if metrics['average_time'] > 60:  # Si toma más de 1 minuto por página en promedio
            logger.warning(f"Tiempo de ejecución alto para {region}. Considerando optimizar el proceso de scraping.")
            # Aquí podrías implementar lógica para optimizar el proceso, como reducir PAGES_PER_REGION

    def log_overall_performance(self):
        logger.info("Resumen de rendimiento:")
        for region, metrics in self.performance_metrics.items():
            logger.info(f"{region}: Tasa de éxito: {metrics['success_rate']:.2f}, Tiempo promedio: {metrics['average_time']:.2f} segundos")