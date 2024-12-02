from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
import openai
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Ensure the OpenAI API key is loaded from the environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Store conversation history
conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history
    user_message = request.json.get('message', '')

    # Add user message to the conversation history
    conversation_history.append({"role": "user", "content": user_message})

    # OpenAI API call
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history,
        )
        bot_response = response.choices[0].message.content

        # Add bot response to the conversation history
        conversation_history.append({"role": "assistant", "content": bot_response})
    except Exception as e:
        bot_response = f"Error: {e}"

    return jsonify({"response": bot_response})

if __name__ == '__main__':
    # Run Flask on localhost
    app.run(host='127.0.0.1', port=5000, debug=True)
