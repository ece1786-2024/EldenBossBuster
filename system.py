from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
import openai


class RAG_agent:
    def __init__(self, index_path):
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        llm = OpenAI(
            model="gpt-4o",
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0.2,
            num_outputs=1000,
        )
        self.query_engine = index.as_query_engine(response_mode="refine", similarity_top_k=5, verbose = True, llm = llm)

    def query(self, query):
        return self.query_engine.query(query)


class Query_agent:
    # This agent takes user prompt and refine to a query for specific RAG agent
    def __init__(self, description, history):
        self.client = openai.OpenAI()
        self.history = history
        self.description = [
            {"role": "system", "content": description},
        ]

    def create_query(self):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=self.history + self.description,
            max_tokens=1000,
            n=1,
            top_p=0.95,
            frequency_penalty=0.2,
            temperature=0.7
        )
        return response.choices[0].message.content
        

class Response_agent:
    # This agent takes the history and create the final response to user
    def __init__(self, description, history):
        self.client = openai.OpenAI()
        self.messages = history
        self.description = [
            {"role": "system", "content": description},
        ]

    def respond(self):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=self.messages + self.description,
            max_tokens=1000,
            n=1,
            top_p=0.95,
            frequency_penalty=0.2,
            temperature=0.7
        )
        return response.choices[0].message.content

KEYWORD = "ADDITIONAL_QUERY_REQUIRED"

class EldenGuideSystem:
    # The agent system that takes user prompt and return the final response
    def __init__(self, strategy_agent, gameinfo_agent, 
                 strategy_query_description, 
                 gameinfo_query_description, 
                 loop_response_description, 
                 final_response_description):
        self.messages = []
        self.strategy_agent = strategy_agent
        self.gameinfo_agent = gameinfo_agent
        self.strategy_query_agent = Query_agent(strategy_query_description, self.messages)
        self.gameinfo_query_agent = Query_agent(gameinfo_query_description, self.messages)
        self.loop_response_agent = Response_agent(loop_response_description, self.messages)
        self.final_response_agent = Response_agent(final_response_description, self.messages)

    def push_assistant_message(self, message):
        self.messages.append({
            "role": "assistant",
            "content": message
        })

    def clear_messages(self):
        self.messages = []

    def run(self, prompt):
        count = 0
        self.messages.append({"role": "user", "content": prompt})
        while count < 3:
            print(f"Querying... {count}")
            strategy_query = self.strategy_query_agent.create_query()
            strategy_response = self.strategy_agent.query(strategy_query)
            self.push_assistant_message("Strategy query result: " + str(strategy_response))
            gameinfo_query = self.gameinfo_query_agent.create_query()
            gameinfo_response = self.gameinfo_agent.query(gameinfo_query)
            self.push_assistant_message("In-game data query result: " + str(gameinfo_response))
            loop_response = self.loop_response_agent.respond()
            if KEYWORD in loop_response:
                count += 1
            else:
                break

        if count >= 3:
            final_response = self.final_response_agent.respond()
        else:
            final_response = loop_response
        return final_response


