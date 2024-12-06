import os
import time
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get('PINECONE_API_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create Pinecone client instance
pc = Pinecone(api_key)

index_name = "apgraderindex"
dimensions = 1536
spec = ServerlessSpec(cloud='aws', region='us-east-1')

# Delete the index if it exists
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

# Create the index
pc.create_index(
    name=index_name,
    dimension=dimensions,
    metric="cosine",
    spec=spec
)

# Wait for the index to be ready
while not pc.describe_index(index_name).status.get('ready', False):
    time.sleep(1)

index = pc.Index(index_name)

# Set OpenAI API key
openai.api_key = openai_api_key

# Construct absolute path to leq.pdf based on this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "leq.pdf")

# Extract text from PDF
reader = PdfReader(pdf_path)
texts = "".join(page.extract_text() for page in reader.pages)

# Create embeddings
response = openai.Embedding.create(input=[texts], model="text-embedding-ada-002")
embedding = response['data'][0]['embedding']

# Upsert vectors to Pinecone
index.upsert([("leq_pdf", embedding, {"text": texts})])

print("Successful")
