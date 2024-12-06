import os
from dotenv import load_dotenv
import openai
from pinecone import Pinecone, ServerlessSpec
from langchain.agents import Tool, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from .pineconesetup import pc, index_name  # Assuming you can import pc and index_name from pineconesetup

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

openai.api_key = OPENAI_API_KEY

# Use the existing Pinecone client (pc) and index
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

def get_relevant_documents(query):
    response = openai.Embedding.create(input=[query], model="text-embedding-ada-002")
    query_embedding = response['data'][0]['embedding']
    results = index.query(query_embedding, top_k=5, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]

prompt = PromptTemplate.from_template("""Your Prompt Template Here""")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4"
)

tools = [
    Tool(
        name="get rubric and sample essays",
        func=lambda query: "\n\n".join(get_relevant_documents(query)),
        description="Use this tool to retrieve relevant sections of the rubric and example essays for grading."
    )
]

agent = initialize_agent(
    llm=llm,
    tools=tools,
    agent="zero-shot-react-description", 
    verbose=True
)

def evaluate_essay(student_essay):
    query = "the entire AP US History LEQ rubric and sample essays"
    relevant_docs = "\n\n".join(get_relevant_documents(query))

    formatted_prompt = prompt.format(
        relevant_docs=relevant_docs,
        student_essay=student_essay
    )

    return agent.run(formatted_prompt)
