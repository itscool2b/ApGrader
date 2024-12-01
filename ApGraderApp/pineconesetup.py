import os
import time
from dotenv import load_dotenv
import pinecone
from PyPDF2 import PdfReader
import openai

# Load environment variables
load_dotenv()

# Get API keys from environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment='us-east-1-aws')

# Index configuration
index_name = "apgraderindex"
dimensions = 1536

# Reset and recreate the index
def reset_index():
    if index_name in pinecone.list_indexes():
        print(f"Deleting existing index: {index_name}")
        pinecone.delete_index(index_name)
        while index_name in pinecone.list_indexes():
            time.sleep(1)

    print(f"Creating index: {index_name}")
    pinecone.create_index(name=index_name, dimension=dimensions, metric="cosine")
    print(f"Index {index_name} created.")

# Wait for index readiness
def wait_for_index_ready(timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        index_description = pinecone.describe_index(index_name)
        if index_description.status == 'ready':
            print(f"Index {index_name} is ready.")
            return
        time.sleep(1)
    raise TimeoutError("Index did not become ready in time.")

# Ingest data into the index
def ingest_data():
    index = pinecone.Index(index_name)

    # Load and read the PDF
    pdf_path = os.path.join(os.path.dirname(__file__), 'leq.pdf')
    try:
        reader = PdfReader(pdf_path)
        texts = "".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return

    # Create embeddings using OpenAI
    try:
        response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
        embedding = response['data'][0]['embedding']
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return

    # Upsert data into the Pinecone index
    try:
        index.upsert([("leq_pdf", embedding, {"text": texts})])
        print("Data successfully ingested.")
    except Exception as e:
        print(f"Error upserting data: {e}")

# Initialize the setup
reset_index()
wait_for_index_ready()
ingest_data()

index = pinecone.Index(index_name)
