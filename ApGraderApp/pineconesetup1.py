import os
import time
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from openai import OpenAI

# Load API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("API keys for Pinecone and OpenAI must be set in the environment variables.")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Configuration
index_name = "apgraderindex"
dimensions = 1536
spec = ServerlessSpec(cloud="aws", region="us-east-1")


def initialize_index():
    """
    Ensures the Pinecone index is created and ready.
    Deletes the index if it already exists for a clean setup.
    """
    # Check if the index exists
    if index_name in pc.list_indexes().names():
        print(f"Index '{index_name}' already exists. Deleting...")
        pc.delete_index(index_name)

    # Create the index
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=dimensions,
        metric="cosine",
        spec=spec
    )

    # Wait for the index to be ready
    print("Waiting for index to be ready...")
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

    print(f"Index '{index_name}' is ready.")


def process_pdf_and_upload(pdf_path):
    """
    Processes a PDF file, generates embeddings, and uploads them to Pinecone.
    """
    # Read and extract text from the PDF
    print(f"Processing PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    texts = "".join([page.extract_text() for page in reader.pages])

    # Generate embeddings using OpenAI
    print("Generating embeddings...")
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    embedding = response.data[0].embedding

    # Upload to Pinecone index
    print("Uploading embeddings to Pinecone...")
    index = pc.Index(index_name)
    index.upsert([("leq_pdf", embedding, {"text": texts})])
    print("Upload successful!")


if __name__ == "__main__":
    # Ensure the Pinecone index is initialized
    initialize_index()

    # Process the PDF and upload embeddings
    pdf_path = "leq.pdf"
    if os.path.exists(pdf_path):
        process_pdf_and_upload(pdf_path)
    else:
        print(f"Error: File '{pdf_path}' not found.")
