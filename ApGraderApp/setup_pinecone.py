import os
import time
import openai
from PyPDF2 import PdfReader
from dotenv import load_dotenv, dotenv_values
from ApGraderApp.p import pc, setup_index, get_index
from openai import OpenAI

# Load environment variables
load_dotenv(dotenv_path="C:/Users/itsco/Desktop/ApGraderProj/ApGraderApp/.env")

# Setup OpenAI and Pinecone
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)
setup_index()

# Wait for the Pinecone index to be ready
while not pc.describe_index("apgraderindex").status.get('ready', False):
    time.sleep(1)

index = get_index()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the folder containing PDFs
pdf_folder = os.path.join(BASE_DIR, "pdf_folder")

# Ensure the pdf_folder exists
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)  # Create the folder if it doesn't exist
    print(f"Folder '{pdf_folder}' created. Please add your PDFs and run the script again.")
    exit()

# Iterate through all PDFs in the folder
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, pdf_file)

        # Read and extract text from the PDF
        reader = PdfReader(pdf_path)
        texts = "".join([page.extract_text() for page in reader.pages])

        # Generate embeddings for the extracted text
        response = client.embeddings.create(
            input=[texts],
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding

        # Use a unique key for each document
        document_key = os.path.splitext(pdf_file)[0]  # Use the file name without the extension as the key

        # Upsert the document into the index
        index.upsert([(document_key, embedding, {"text": texts})])

print("Index setup complete")
