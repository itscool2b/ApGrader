import os
import time
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from openai import OpenAI
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
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)

client = OpenAI(api_key=openai_api_key)

# Extract text from PDF
reader = PdfReader("leq.pdf")
texts = "".join([page.extract_text() for page in reader.pages])

# Create embeddings
response = client.embeddings.create(input=texts, model="text-embedding-ada-002")
embedding = response.data[0].embedding

# Upsert vectors to Pinecone
index.upsert([("leq_pdf", embedding, {"text": texts})])

print("Succecful")
