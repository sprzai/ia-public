from src.agents.english_course_agent import create_english_course_agent, interact_with_agent, generate_english_course
import argparse

def main():
    """
    Función principal para interactuar con el agente del curso de inglés.
    """
    parser = argparse.ArgumentParser(description="Interact with English Course Agent")
    parser.add_argument("--llm", choices=["anthropic", "llama"], default="anthropic", help="LLM to use (anthropic or llama)")
    parser.add_argument("--topic", type=str, default="General English", help="Topic for the English course")
    args = parser.parse_args()

    # Crear el agente
    agent = create_english_course_agent(args.llm)

    # Generar el curso de inglés
    course = generate_english_course(agent, args.topic)

    # Imprimir un resumen del curso generado
    print(f"Generated a 30-day English course on {args.topic}")
    for lesson in course:
        print(f"Lesson {lesson['lesson_number']}:")
        print(f"  Dialogue: {lesson['dialogue'][:50]}...")
        print(f"  Text: {lesson['text'][:50]}...")
        print(f"  Vocabulary: {len(lesson['vocabulary'])} words")
        print()

    # Interacción adicional con el agente
    while True:
        user_input = input("Ask a question about the course (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        response = interact_with_agent(agent, user_input)
        print("Agent response:", response)

if __name__ == "__main__":
    main()




# import os
# from dotenv import load_dotenv
# from src.agents.english_course_agent import EnglishCourseAgent
# def main():
#     load_dotenv()
#     anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
#     whatsapp_number = os.getenv("WHATSAPP_NUMBER")
#     morning_time = os.getenv("MORNING_TIME", "08:00")
#     afternoon_time = os.getenv("AFTERNOON_TIME", "16:00")
#     agent = EnglishCourseAgent(anthropic_api_key, whatsapp_number)
#     agent.initialize_schedule(morning_time, afternoon_time)
#     agent.run()
# if __name__ == "__main__":
#     main()
