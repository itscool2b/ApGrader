import os
from dotenv import load_dotenv
import openai
import pinecone
from langchain.agents import Tool, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1-aws")

openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

index_name = "apgraderindex"
index = pinecone.Index(index_name)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

def get_relevant_documents(query):
    response = openai.Embedding.create(input=[query], model="text-embedding-ada-002")
    query_embedding = response['data'][0]['embedding']
    results = index.query(query_embedding, top_k=5, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]

prompt = PromptTemplate.from_template("""Your prompt template here...""")

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
