import logging
import os
import requests
import time
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain.tools.retriever import create_retriever_tool
from langchain.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# --- STATE & MEMORY ---
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# =====================================================================
# GLOBAL SESSION STORE (In-Memory for Development; swap for Redis/SQL in Prod)
# =====================================================================
message_history_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Retrieves or initializes a isolated chat history for a specific session ID."""
    if session_id not in message_history_store:
        logging.info(f"Creating a completely fresh conversational memory session for ID: {session_id}")
        message_history_store[session_id] = InMemoryChatMessageHistory()
    return message_history_store[session_id]


def get_agent_executor():
    """
    Creates, wraps, and returns a stateful LangChain AgentExecutor.
    """
    logging.info("--- Initializing Agent Logic ---")

    # 1. Initialize LLM
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.5)

    # 2. Setup RAG Knowledge Base
    try:
        food_loader = TextLoader("./knowledge_base/lucknow_food.txt")
        history_loader = TextLoader("./knowledge_base/lucknow_history.txt")
        documents = food_loader.load() + history_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        logging.info("Using Hugging Face Inference API (BAAI/bge-small-en-v1.5) to avoid OOM.")
        embedding_model = HuggingFaceEndpointEmbeddings(model="BAAI/bge-small-en-v1.5")
        persist_dir = "./chroma_db_lucknow"

        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            logging.info("Loading existing persistent Chroma database from disk...")
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
        else:
            logging.info("Vector DB not found. Generating embeddings in safe batches...")
            # Initialize an empty vector store
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
            
            # ---  BATCHING FOR RATE LIMITS ---
            # Instead of crashing the free-tier API, we upload in safe batches with a delay
            batch_size = 20
            for i in range(0, len(splits), batch_size):
                batch = splits[i:i + batch_size]
                vectorstore.add_documents(batch)
                logging.info(f"Ingested batch {i} to {i + len(batch)}...")
                time.sleep(1.5) # Protects against 429 Too Many Requests errors
            
        # ---  MAXIMAL MARGINAL RELEVANCE (MMR) ---
        # Standard search returns redundant data. MMR fetches 15 chunks, 
        # then algorithmically selects the 4 most *diverse* ones to give the LLM better context.
        retriever = vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 4, "fetch_k": 15}
        )
    except Exception as e:
        logging.error(f"Failed to create vector store: {e}")
        return None

    # 3. Define Tools
    retriever_tool = create_retriever_tool(
        retriever,
        "lucknow_knowledge_base",
        "PRIMARY SOURCE. Access this tool first for any inquiries regarding Lucknow's food, history, "
        "culture, historical monuments, traditional markets, and curated local guides or itineraries."
    )

    if os.getenv("TAVILY_API_KEY"):
        web_search = TavilySearchResults(max_results=2)
        web_search.description = (
            "FALLBACK WEB SEARCH. Use this tool ONLY if the 'lucknow_knowledge_base' yields no results, "
            "or if the user explicitly asks for real-time information such as newly opened venues, "
            "current hotel listings, or live events."
        )
    else:
        @tool
        def web_search(query: str) -> str:
            """FALLBACK WEB SEARCH. Use only if primary local knowledge base yields no results."""
            return "Live web search currently unavailable due to system maintenance."

    @tool
    def weather_tool(city: str = "Lucknow") -> str:
        """Fetches the real-time weather for a specified city. Mandatorily execute this tool "
        "whenever you need current weather conditions to tailor active travel schedules or daily itineraries."""
        latitude, longitude = 26.8467, 80.9462
        BASE_URL = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": latitude, "longitude": longitude, "current_weather": "true"}
        
        try:
            response = requests.get(BASE_URL, params=params, timeout=5)
            response.raise_for_status()
            data = response.json().get('current_weather', {})
            return f"The current temperature in Lucknow is {data.get('temperature', 'N/A')}°C with a wind speed of {data.get('windspeed', 'N/A')} km/h."
        except Exception:
            return "Unable to fetch live weather at this time. Proceed with general travel advice."

    tools = [retriever_tool, weather_tool, web_search]

    # 4. Declarative System Prompt with Chat History Placeholder
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an elite AI Travel Concierge specializing exclusively in Lucknow, India. "
            "You are meticulous, factual, and highly disciplined in your use of data.\n\n"
            
            "### CORE INTERACTION POLICIES\n"
            "- GREETINGS: If the user interactions are simple conversational greetings (e.g., 'hi', 'hello', 'good morning'), "
            "respond warmly, introduce your specialty, and ask how you can help. DO NOT execute tools or build itineraries for generic greetings.\n"
            "- ITINERARY MANDATE: You are only permitted to generate a structured, day-by-day itinerary if the user explicitly "
            "asks for a trip plan, schedule, or itinerary.\n"
            "- CONTEXT AWARENESS: You have access to past user turns in the chat history. Read the context to seamlessly handle follow-up "
            "requests, adjustments, modifications, or clarifications (e.g., shortening a previously given itinerary).\n"
            "- OUT OF SCOPE: Politely decline to answer any questions completely unrelated to travel or Lucknow.\n\n"

            "### DATA CONSTRAINTS & ZERO-HALLUCINATION\n"
            "- You operate under a strict zero-hallucination mandate. Rely entirely on the information returned from your tool outputs.\n"
            "- If a tool indicates a location exists but lacks metadata details (such as ambiance, specific dishes, or prices), "
            "do not invent or extrapolate those descriptions. If it is not in the tool text, you do not know it.\n"
            "- Rely on the local knowledge base for regional history and cuisine before using secondary fallback layers.\n"
            "- When building explicitly requested itineraries, query your weather tool to incorporate an active, practical weather advisory point.\n\n"

            "### SOURCE LINEAGE & TAGGING\n"
            "- For any specific recommendation, place, or fact retrieved from the web search tool, you MUST "
            "immediately append this exact phrase after that specific item: *(Verified via live web update and not part of the verified local Lucknow guide)*.\n"
            "- Maintain strict data separation. Do not combine or blend facts from the local knowledge base and the web search inside the same sentence structure."
        )),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 5. Create the Base Agent Execution Layer
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    base_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    # 6. Wrap the execution layer inside stateful runnable history pipeline
    stateful_agent_executor = RunnableWithMessageHistory(
        base_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    
    return stateful_agent_executor
