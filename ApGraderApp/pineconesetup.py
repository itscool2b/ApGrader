# pineconesetup.py

import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
import openai  # Import the OpenAI library

# Load environment variables
load_dotenv()

# Get API keys from environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Create an instance of the Pinecone class
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index configuration
index_name = "apgraderindex"
dimensions = 1536
spec = ServerlessSpec(cloud='aws', region='us-east-1')

# Function to reset and recreate the index
def reset_index():
    if index_name in pc.list_indexes().names():
        print(f"Deleting existing index: {index_name}")
        pc.delete_index(index_name)
        # Wait until the index is deleted
        while index_name in pc.list_indexes().names():
            time.sleep(1)

    print(f"Creating index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=dimensions,
        metric="cosine",
        spec=spec
    )
    print(f"Index {index_name} created.")

# Wait for the index to be ready
def wait_for_index_ready(timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = pc.describe_index(index_name).status
        if status.get("ready"):
            print(f"Index {index_name} is ready.")
            return
        time.sleep(1)
    raise TimeoutError(f"Index {index_name} did not become ready within {timeout} seconds.")

# Function to ingest data into the index
def ingest_data():
    # Connect to the Pinecone index
    index = pc.Index(index_name)
    
    # Read the PDF
    try:
        pdf_path = os.path.join(os.path.dirname(__file__), 'leq.pdf')
        reader = PdfReader(pdf_path)
        texts = "".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return

    # Create embeddings using the new OpenAI API
    try:
        response = openai.Embedding.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        embedding = response['data'][0]['embedding']
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return

    # Upsert data into the index
    try:
        index.upsert([("leq_pdf", embedding, {"text": texts})])
        print("Data successfully ingested into the index.")
    except Exception as e:
        print(f"Error upserting data: {e}")

# Execute the initialization
reset_index()
wait_for_index_ready()
ingest_data()

# Export the index for use in other modules
index = pc.Index(index_name)
