import logging
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import AgentExecutor
from typing import List, Union, Dict
import faiss
import os
from dotenv import load_dotenv
from langchain_community.chat_models.ollama import ChatOllama

# Cargar variables de entorno
load_dotenv()

# Configuración del logging
logging.basicConfig(filename='agent_detailed.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Definición del template de prompt
prompt_template = """Eres un asistente útil que crea un curso de inglés personalizado.

Tienes acceso a las siguientes herramientas:
{tools}

Usa el siguiente formato:

Pregunta: la pregunta del usuario
Pensamiento: siempre debes pensar sobre qué hacer
Acción: la acción a tomar, debe ser una de [{tool_names}]
Entrada de acción: la entrada a la acción
Observación: el resultado de la acción
... (este Pensamiento/Acción/Entrada de acción/Observación puede repetirse N veces)
Pensamiento: Ahora sé la respuesta final
Respuesta final: la respuesta final a la pregunta del usuario

Comienza:
Pregunta: {input}
Pensamiento: {agent_scratchpad}"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
)

def create_english_course_agent(llm_type: str = "anthropic") -> AgentExecutor:
    """
    Crea y configura el agente para el curso de inglés.
    
    Args:
        llm_type (str): Tipo de LLM a utilizar ('anthropic' o 'llama').
    
    Returns:
        AgentExecutor: El agente ejecutor configurado.
    """
    logging.info(f"Iniciando la creación del agente de curso de inglés con LLM: {llm_type}")
    
    # Selección del LLM
    # if llm_type.lower() == "anthropic":
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logging.error("ANTHROPIC_API_KEY no está configurada en las variables de entorno.")
        raise ValueError("ANTHROPIC_API_KEY no está configurada en las variables de entorno.")
    llm = ChatAnthropic(
        temperature=0,
        anthropic_api_key=anthropic_api_key,
        model_name="claude-3-5-sonnet-20240620"
    )
    logging.info("LLM Anthropic (Claude-2) configurado correctamente")
    # elif llm_type.lower() == "llama":
    # llm = ChatOllama(model="llama3", temperature=0)
    # logging.info("LLM Llama configurado correctamente")
    # else:
    #     logging.error(f"Tipo de LLM no válido: {llm_type}")
    #     raise ValueError("LLM type must be 'anthropic' or 'llama'")
    
    logging.info("Configurando el almacenamiento vectorial")
    # Configuración del almacenamiento vectorial
    embeddings = HuggingFaceEmbeddings()
    embedding_size = 384 
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings.embed_query, index, InMemoryDocstore({}), {})
    logging.info("Almacenamiento vectorial configurado correctamente")
    
    # Herramientas del agente
    logging.info("Definiendo las herramientas del agente")
    tools = [
        Tool(
            name="Vector Store",
            func=vectorstore.similarity_search,
            description="Útil para buscar información relevante en el almacenamiento vectorial."
        )
    ]
    logging.info(f"Herramientas definidas: {[tool.name for tool in tools]}")
    
    # Configuración de la memoria
    logging.info("Configurando la memoria del agente")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Creación del agente utilizando initialize_agent
    logging.info("Inicializando el agente")
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        prompt=prompt
    )
    logging.info("Agente inicializado correctamente")
    
    return agent

def interact_with_agent(agent: AgentExecutor, user_input: str) -> str:
    """
    Interactúa con el agente y registra la interacción.
    
    Args:
        agent (AgentExecutor): El agente ejecutor.
        user_input (str): La entrada del usuario.
    
    Returns:
        str: La respuesta del agente.
    """
    logging.info(f"Iniciando interacción con el agente. Entrada del usuario: {user_input}")
    try:
        logging.debug("Ejecutando el agente")
        response = agent.run(user_input)
        logging.info(f"Respuesta del agente obtenida: {response[:100]}...")  # Log primeros 100 caracteres
        return response
    except Exception as e:
        logging.error(f"Error durante la interacción con el agente: {str(e)}")
        return f"An error occurred: {str(e)}"

def generate_english_course(agent: AgentExecutor, topic: str) -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
    """
    Genera un curso de inglés de 30 clases utilizando el agente.
    
    Args:
        agent (AgentExecutor): El agente ejecutor.
        topic (str): El tema del curso de inglés.
    
    Returns:
        List[Dict[str, Union[str, List[Dict[str, str]]]]]: Lista de 30 clases de inglés.
    """
    logging.info(f"Iniciando la generación del curso de inglés sobre: {topic}")
    course = []
    for i in range(30):
        logging.info(f"Generando lección {i+1} de 30")
        prompt = f"Generate English lesson {i+1} about {topic} with a dialogue, text, and vocabulary (English, Spanish, pronunciation)."
        response = interact_with_agent(agent, prompt)
        
        logging.debug(f"Procesando respuesta para la lección {i+1}")
        # Procesar la respuesta para estructurar la lección
        # (Este es un ejemplo simplificado, se necesitaría un parsing más robusto en un escenario real)
        lesson = {
            "lesson_number": i+1,
            "dialogue": "Example dialogue",
            "text": "Example text",
            "vocabulary": [
                {"english": "word", "spanish": "palabra", "pronunciation": "wərd"}
            ]
        }
        course.append(lesson)
        logging.info(f"Lección {i+1} agregada al curso")
    
    logging.info("Generación del curso completada")
    return course

if __name__ == "__main__":
    logging.info("Iniciando el programa principal")
    agent = create_english_course_agent("llama")  # anthropic
    topic = "Business English"
    logging.info(f"Generando curso de inglés sobre: {topic}")
    course = generate_english_course(agent, topic)
    logging.info(f"Curso generado exitosamente. Total de lecciones: {len(course)}")
    print(f"Generated a 30-day English course on {topic}")
    logging.info("Programa finalizado")