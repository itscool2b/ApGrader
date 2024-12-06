import os
import time
import openai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pineconesetup import pc, setup_index, get_index

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Run this script manually once or on a deploy hook
setup_index()

# Wait for index to be ready
while not pc.describe_index("apgraderindex").status.get('ready', False):
    time.sleep(1)

index = get_index()

# Extract text from PDF
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "ApGraderApp", "leq.pdf")
reader = PdfReader(pdf_path)
texts = "".join([page.extract_text() for page in reader.pages])

# Create embeddings
response = openai.Embedding.create(input=[texts], model="text-embedding-ada-002")
embedding = response['data'][0]['embedding']

# Upsert vectors to Pinecone
index.upsert([("leq_pdf", embedding, {"text": texts})])

print("Index setup complete")
