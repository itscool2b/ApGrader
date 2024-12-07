import os
import time
import openai
from PyPDF2 import PdfReader
from dotenv import load_dotenv, dotenv_values
from pineconesetup import pc, setup_index, get_index


load_dotenv(dotenv_path="C:/Users/itsco/Desktop/ApGraderProj/ApGraderApp/.env")

openai.api_key = os.getenv("OPENAI_API_KEY")

setup_index()


while not pc.describe_index("apgraderindex").status.get('ready', False):
    time.sleep(1)

index = get_index()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "leq.pdf")

reader = PdfReader(pdf_path)
texts = "".join([page.extract_text() for page in reader.pages])

response = openai.Embedding.create(
    input=[texts],
    model="text-embedding-ada-002"
)
embedding = response["data"][0]["embedding"]

index.upsert([("leq_pdf", embedding, {"text": texts})])


print("Index setup complete")
