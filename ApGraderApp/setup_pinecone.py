import os
import time
import openai
from PyPDF2 import PdfReader
from dotenv import load_dotenv, dotenv_values
from p import pc, setup_index, get_index
from openai import OpenAI


load_dotenv(dotenv_path="C:/Users/itsco/Desktop/ApGraderProj/ApGraderApp/.env")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)
setup_index()


while not pc.describe_index("apgraderindex").status.get('ready', False):
    time.sleep(1)

index = get_index()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


pdf_folder = os.path.join(BASE_DIR, "pdf_folder")


if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)  
    print(f"Folder '{pdf_folder}' created. Please add your PDFs and run the script again.")
    exit()


for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, pdf_file)

        
        reader = PdfReader(pdf_path)
        texts = "".join([page.extract_text() for page in reader.pages])

       
        response = client.embeddings.create(
            input=[texts],
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding

        
        document_key = os.path.splitext(pdf_file)[0]  

        
        index.upsert([(document_key, embedding, {"text": texts})])

print("Index setup complete")
