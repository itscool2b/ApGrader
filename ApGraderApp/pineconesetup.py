import os
import time
import pinecone
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1")

# Index configuration
index_name = "apgraderindex"
dimensions = 1536

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to reset and recreate the index
def reset_index():
    if index_name in pinecone.list_indexes():
        print(f"Deleting existing index: {index_name}")
        pinecone.delete_index(index_name)

    print(f"Creating index: {index_name}")
    pinecone.create_index(name=index_name, dimension=dimensions, metric="cosine")
    print(f"Index {index_name} created.")

# Wait for the index to be ready
def wait_for_index_ready(timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = pinecone.describe_index(index_name).status
        if status.get("ready"):
            print(f"Index {index_name} is ready.")
            return
        time.sleep(1)
    raise TimeoutError(f"Index {index_name} did not become ready within {timeout} seconds.")

# Function to ingest data into the index
def ingest_data():
    # Connect to the Pinecone index
    index = pinecone.Index(index_name)
    
    # Read the PDF
    try:
        reader = PdfReader("leq.pdf")
        texts = "".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return

    # Create embeddings
    try:
        response = client.embeddings.create(input=texts, model="text-embedding-ada-002")
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

# Main block to control execution
if __name__ == "__main__":
    reset_index()
    wait_for_index_ready()
    ingest_data()
    print("Setup completed successfully.")
