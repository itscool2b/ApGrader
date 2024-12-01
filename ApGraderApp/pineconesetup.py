import os
import time
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader

from openai import OpenAI

api_key = os.environ.get('PINECONE_API_KEY')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key)

index_name = "apgraderindex"

dimensions = 1536
spec = ServerlessSpec(cloud='aws',region='us-east-1')

if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

pc.create_index(name=index_name,
                dimension=dimensions,
                metric="cosine",
                spec=spec)




while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)

#reader = PdfReader("leq.pdf")
#texts = "".join([page.extract_text() for page in reader.pages])

#response = client.embeddings.create(input=texts, model="text-embedding-3-small")
#embedding = response.data[0].embedding
#index.upsert([("leq_pdf", embedding, {"text": texts})])

#print("Succecful")
result = index.query(queries=["leq_pdf"], top_k=1, include_metadata=True)

if result["matches"]:
    print("Data is already in Pinecone:", result["matches"][0]["metadata"]["text"])
else:
    print("Data not found in Pinecone.")