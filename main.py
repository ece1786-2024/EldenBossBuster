from system import EldenGuideSystem, RAG_agent
import os
from dotenv import load_dotenv
import openai
import time
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def load_prompt(filename):
        with open(f'agent_descriptions/{filename}.txt', 'r') as file:
            return file.read().strip()
        
def greet():
    return (
        "Welcome to the LLM Multi-Agent System Chat Interface.\n"
        "Type 'exit' or 'quit' to end the chat.\n"
        "Type 'new' to start a new chat.\n"
        "Type 'hidden' to see the hidden messages.\n"
    )

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
    strategy_path = "index\strategy"
    gameinfo_path = "index\game"
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

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/chat', methods=['POST'])
def chat():
    global system
    user_message = request.json.get('message', '').strip().lower()

    if user_message == 'new':
        system.clear_messages()
        # Return the greeting message here
        return jsonify({"response": greet()})
    elif user_message == 'hidden':
        return jsonify({"response": f"Hidden messages: {system.messages}"})

    # Normal message handling
    try:
        response = system.run(user_message)
    except Exception as e:
        response = f"Error: {e}"

    return jsonify({"response": response})

if __name__ == "__main__":
    system = init()  # Initialize the system once
    # Run the Flask server
    app.run(host='127.0.0.1', port=5000, debug=True)
