import os
import time
import openai
from PyPDF2 import PdfReader
from dotenv import load_dotenv, dotenv_values
from pineconesetup import pc, setup_index, get_index


# Ensure this matches your .env key name
load_dotenv(dotenv_path="C:/Users/itsco/Desktop/ApGraderProj/ApGraderApp/.env")
print("After specifying path:", os.getenv("OPENAI_API_KEY")) # This loads .env from the same directory as manage.py (if you're running from there)
print("Env var directly:", os.environ.get("OPENAI_API_KEY"))
# Print to verify what key we got
print("OPENAI_API_KEY from .env:", os.getenv("OPENAI_API_KEY"))

# Set the key for OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Print to confirm the final key used by openai
print("openai.api_key set to:", openai.api_key)

# Run this script manually or via a deploy hook to set up the index first
setup_index()

# Wait until Pinecone index is ready
while not pc.describe_index("apgraderindex").status.get('ready', False):
    time.sleep(1)

index = get_index()

# Construct the path to the PDF
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "leq.pdf")

reader = PdfReader(pdf_path)
texts = "".join([page.extract_text() for page in reader.pages])

# Use openai.Embedding.create as per pre-1.0.0 API (works in 0.28.0)
response = openai.Embedding.create(
    input=[texts],
    model="text-embedding-ada-002"
)
embedding = response["data"][0]["embedding"]

# Upsert to Pinecone
index.upsert([("leq_pdf", embedding, {"text": texts})])


# Upsert vectors to Pinecone


print("Index setup complete")
