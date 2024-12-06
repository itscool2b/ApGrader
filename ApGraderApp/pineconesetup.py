import os
import time
import pinecone
from PyPDF2 import PdfReader
import openai
from dotenv import load_dotenv

load_dotenv()

# Set API keys
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone_env = os.environ.get('PINECONE_ENV', 'us-east-1-aws')
openai_api_key = os.environ.get('OPENAI_API_KEY')

openai.api_key = openai_api_key

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

index_name = "apgraderindex"
dimensions = 1536

# Delete index if exists
if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

# Create index
pinecone.create_index(
    name=index_name,
    dimension=dimensions,
    metric="cosine"
)

# Wait until index is ready
while True:
    desc = pinecone.describe_index(index_name)
    if desc.status.get('ready', False):
        break
    time.sleep(1)

index = pinecone.Index(index_name)

reader = PdfReader("leq.pdf")
texts = "".join([page.extract_text() for page in reader.pages])

response = openai.Embedding.create(input=[texts], model="text-embedding-ada-002")
embedding = response['data'][0]['embedding']

index.upsert([("leq_pdf", embedding, {"text": texts})])

print("Successful")
