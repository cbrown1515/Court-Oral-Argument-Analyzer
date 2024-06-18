import os 
import dotenv
import logging
from openai import Client as OpenAIClient 
from qdrant_client import QdrantClient
from langchain_openai import OpenAI as LangchainOpenAI 
from llama_index.core.llms import LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.storage import StorageContext
import chainlit as cl
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.settings import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.cohere import Cohere
from llama_index.postprocessor.colbert_rerank import ColbertRerank 
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
import qdrant_client
#from langsmith.wrappers import wrap_openai
from langsmith import traceable
#from langsmith.evaluation import evaluate
from langchain.schema.runnable.config import RunnableConfig
from llama_index.llms.nvidia import NVIDIA
from llama_index.core import PromptTemplate
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank
from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


dotenv.load_dotenv()
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#s.environ["LANGCHAIN_API_KEY"] =os.getenv("LANGCHAIN_API_KEY")
api_key=os.getenv("QDRANT_API_KEY")
import os

# del os.environ['NVIDIA_API_KEY']  ## delete key and reset
if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
    nvapi_key = os.getenv("NVIDIA_API_KEY")
    if nvapi_key and nvapi_key.startswith("nvapi-"):
        print("NVIDIA_API_KEY set from environment variable.")
    else:
        raise ValueError("NVIDIA_API_KEY environment variable is not set or invalid.")
embedding = OpenAIEmbedding()
collection_name="OralARGSPOC"

#Langsmith auto-trace LLM calls in-context
#openai_client = OpenAIClient()
#client = wrap_openai(openai_client)

# Qdrant Client Configuration
qdrant_collection_name = "OralARGSPOC"
client = qdrant_client.QdrantClient(
            "https://f7b73111-3914-4ee3-82fb-611a5343d972.us-east-1-0.aws.cloud.qdrant.io",
            api_key=os.getenv("QDRANT_API_KEY"), # For Qdrant Cloud, None for local instance
        )

vector_store = QdrantVectorStore(
            collection_name='OralARGSPOC',
            client=client,
            #aclient=aclient,
            enable_hybrid=True,
            batch_size=20,
        )
vector_store = QdrantVectorStore(client=client, collection_name=qdrant_collection_name,enable_hybrid=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store,
                                               storage_context=storage_context,batch_size=10)

legal_thinker= NVIDIA(model="meta/llama3-70b-instruct")

@tool 
@traceable(name="Search and Process")
def search_and_process(query: str):
    """
    This function takes a query string as input and processes it using a query engine.
    
    The query engine is configured to use a hybrid vector store query mode.
    
    Args:
        query (str): The query string to be processed.
    
    Returns:
        The result of the query processed by the query engine, including the response and metadata.
    """
    service_context = ServiceContext.from_defaults(callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]))

    logging.info(f'Processing query: {query}')
    query_engine = index.as_query_engine(
        llm=legal_thinker,
        vector_store_query_mode="hybrid", 
        similarity_top_k=3,
        sparse_top_k=10,
        service_context=service_context,
        node_postprocessor=[NVIDIARerank(model='nvidia/rerank-qa-mistral-4b', top_n=10)]
    )

    thinker_template = ("""
       You are an expert system tasked with retrieving and synthesizing information from oral argument transcripts. Your goal is to provide a detailed and accurate response based on the case law and oral arguments available in our database.

    Here is the snippet of the oral argument transcript:
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
                    
    Please synthesize the information in the oral argument transcript and answer the following question:
    Specific Question: 

    Please follow these guidelines to structure your response:
    - Include the full name and citation of any statute, case law, law, or regulation mentioned.
    - Include the full text of any relevant excerpts from the oral argument transcript.
    - Provide a clear and thorough answer to the specific question asked.
    - Ensure that all information is factually correct and up-to-date, drawing from the most reliable legal sources available.
""")
    legal_reason = PromptTemplate(thinker_template)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": legal_reason})
    result = query_engine.query(query)
    sources = []
    for node in result.source_nodes:
        source_text = node.get_text()
        source_metadata = node.metadata
        sources.append({"text": source_text, "metadata": source_metadata})
    logging.info(f'Query result: {result}')
    
    # Append sources to citations here before returning
    
    return {
        "response": result,
        "sources": sources,
    }

context = "Oral arguments"
prompt = """

You are an expert in U.S. Federal Court Cases and Oral Arguments. Your overarching purpose is when given a question about a U.S. Federal Court Case, Oral Argument, or Judge, you are to provide an accurate and complete response utilizing the case law and oral arguments returned as output from the search_and_process function.

For the response always include a Sources section where you return the 'sources' list from the search_and_process function output including any quoted text in the sources.

TLDR: Provide a concise response to User Question in the TLDR section.

Analysis: Provide a detailed analysis of the facts, holding, and any statutes, laws, or regulations at issue in the Analysis section.

Follow the format below to ensure clarity and thoroughness in your answer. Format Example:

User Question:“What statute or statutes were at issue and what was the holding in Haque v. Holder?”
Agent Response:“In the case of Haque v. Holder, the primary statutes at issue were the Antiterrorism and Effective Death Penalty Act (AEDPA) and the Illegal Immigrant Reform and Immigrant Responsibility Act (IIRIRA).

Case: [Case Name with full citation]
Judge(s): [Judge Name] 

TLDR: [short core response to User Question]

Analysis: [detailed analysis of the facts, holding and any statutes, laws or regulations at issue]

Sources: 
1. Oral Argument Excerpt: "[excerpt text]..."[Excerpt from oral argument].
2. Oral Argument Excerpt: "[excerpt text]..."[Excerpt from oral argument].
3. Oral Argument Excerpt: "[excerpt text]..."[Excerpt from oral argument].
4. Oral Argument Excerpt: "[excerpt text]..."[Excerpt from oral argument].
5. Oral Argument Excerpt: "[excerpt text]..."[Excerpt from oral argument].
6. Case Law: "[case law text]..."[Case law citation].

"""
system_message = SystemMessage(
    content=prompt
)
agent_prompt = OpenAIFunctionsAgent.create_prompt(system_message)

llm=ChatOpenAI(model="gpt-3.5-turbo")
agent = OpenAIFunctionsAgent(llm=llm, tools=[search_and_process,], prompt=agent_prompt)

agent_executor = AgentExecutor(agent=agent, tools=[search_and_process],verbose=True)



@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Hello there, I am AI Agent. How can I help you?").send()
    cl.user_session.set("agent", agent_executor)

@cl.on_message
async def on_message(message: cl.Message):
    agent_executor = cl.user_session.get("agent")
    result = await agent_executor.ainvoke(message.content)
    # Extract the 'output' part of the result and send it
    output_text = result.get('output', 'Sorry, I could not process your request.')
    response_message = cl.Message(content=output_text)
    await response_message.send()