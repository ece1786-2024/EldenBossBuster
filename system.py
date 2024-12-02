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
    def __init__(self, description, history=[]):
        self.client = openai.OpenAI()
        self.messages = history + [
            {"role": "system", "content": description},
            {"role": "user", "content": ""}
        ]

    def create_query(self, prompt):
        self.messages[1]["content"] = prompt
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=self.messages,
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
        print(self.messages)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=self.messages,
            max_tokens=1000,
            n=1,
            top_p=0.95,
            frequency_penalty=0.2,
            temperature=0.7
        )
        return response.choices[0].message.content


STRATEGY_QUERY_DESCRIPTION = """You are a Strategy Query Refinement Agent.
Your primary function is to interpret the user's prompts related to 
gameplay strategies, character builds, or weapon recommendations in Elden Ring. 
You focus on understanding the user's goals, preferences, and challenges. You 
then refine and formulate an effective query tailored to retrieve relevant 
information from the "YouTuber Strategy and Weapon Recommendations" vector store. 
By crafting precise queries, you enhance the RAG process, ensuring that the most 
pertinent strategic insights are fetched on behalf of the user.
You should ONLY return a single query itself."""

GAMEINFO_QUERY_DESCRIPTION = """You are an In-Game Data Query Refinement Agent.
Specializing in in-game data retrieval, you analyze the user's requests for 
specific information about items, such as locations, weights, prices, or stats 
within Elden Ring. Your role is to comprehend the exact details the user is 
seeking and refine the query accordingly. You construct an optimized search query 
to access the required information from the "In-Game Data" vector store sourced 
from the Elden Ring wiki. By focusing on precise query formulation, you facilitate 
the RAG process to deliver accurate in-game data on behalf of the user.
You should ONLY return a single query itself."""

FINAL_RESPONSE_DESCRIPTION = """You are an intelligent response agent tasked with 
generating clear and helpful answers to the user's initial prompts by effectively 
utilizing the query results obtained from the RAG process. With access to the 
complete message history—including the user's original question and the information 
retrieved from both the "YouTuber Strategy and Weapon Recommendations" and "In-Game 
Data" vector stores—you analyze the relevance and accuracy of the retrieved 
information. Your goal is to construct informed responses that integrate pertinent 
data to provide comprehensive and useful answers. If the query results are 
irrelevant or insufficient, you recognize this and avoid confusing the user, opting 
instead to inform them that specific information is not available or to ask 
clarifying questions. By intelligently handling both relevant and irrelevant query 
results, you ensure the user receives meaningful assistance without unnecessary 
confusion, enhancing their experience with the Elden Ring game guide system."""


class EldenGuideSystem:
    def __init__(self, strategy_agent, gameinfo_agent):
        self.messages = []
        self.strategy_agent = strategy_agent
        self.gameinfo_agent = gameinfo_agent
        self.strategy_query_agent = Query_agent(STRATEGY_QUERY_DESCRIPTION)
        self.gameinfo_query_agent = Query_agent(GAMEINFO_QUERY_DESCRIPTION)
        self.final_response_agent = Response_agent(FINAL_RESPONSE_DESCRIPTION, self.messages)

    def push_assistant_message(self, message):
        self.messages.append({
            "role": "assistant",
            "content": message
        })

    def run(self, prompt):
        strategy_query = self.strategy_query_agent.create_query(prompt)
        strategy_response = self.strategy_agent.query(strategy_query)
        self.push_assistant_message(strategy_response)
        gameinfo_query = self.gameinfo_query_agent.create_query(prompt)
        gameinfo_response = self.gameinfo_agent.query(gameinfo_query)
        self.push_assistant_message(gameinfo_response)
        final_response = self.final_response_agent.respond()
        return final_response
