from system import EldenGuideSystem, RAG_agent
import os
from dotenv import load_dotenv
import openai
import time
def load_prompt(filename):
        with open(f'agent_descriptions/{filename}.txt', 'r') as file:
            return file.read().strip()
        
def greet():
    print("Welcome to the LLM Multi-Agent System Chat Interface.")
    print("Type 'exit' or 'quit' to end the chat.")
    print("Type 'new' to start a new chat.")
    print("Type 'hidden' to see the hidden messages.\n")

def init():
    # load the api key
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai.api_key
    # Initialize the system
    # load the agent descriptions
    STRATEGY_QUERY_DESCRIPTION = load_prompt('strategy_agent')
    GAMEINFO_QUERY_DESCRIPTION = load_prompt('gameinfo_agent')
    FINAL_RESPONSE_DESCRIPTION = load_prompt('final_agent')
    LOOP_RESPONSE_DESCRIPTION = load_prompt('loop_agent')
    # set up rag agents
    print("Loading RAG vectorstores... [Could take a while (20 mins)]")
    # log time, hour:minute:second  
    print(f"Start time: {time.strftime('%H:%M:%S', time.localtime())}")
    strategy_path = "index/strategy"
    gameinfo_path = "index/game"
    strategy_agent = RAG_agent(strategy_path)
    gameinfo_agent = RAG_agent(gameinfo_path)
    print(f"End time: {time.strftime('%H:%M:%S', time.localtime())}")
    # set up the system
    print("Setting up the system...")
    system = EldenGuideSystem(strategy_agent, gameinfo_agent,
                              STRATEGY_QUERY_DESCRIPTION,
                              GAMEINFO_QUERY_DESCRIPTION,
                              LOOP_RESPONSE_DESCRIPTION,
                              FINAL_RESPONSE_DESCRIPTION)
    return system
        
def main():
    system = init()
    greet()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        elif user_input.lower() == 'new':
            system.clear_messages()
            greet()
            continue
        elif user_input.lower() == 'hidden':
            print(system.messages)
            continue

        # Send the input to the LLM system and get the response
        response = system.run(user_input)

        print(f"LLM: {response}\n")
        print("Type 'exit' or 'quit' to end, 'new' to start a new chat.")

if __name__ == "__main__":
    main()