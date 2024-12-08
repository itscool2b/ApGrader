import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key)

index_name = "apgraderindex"
dimensions = 1536
spec = ServerlessSpec(cloud='aws', region='us-east-1')

def setup_index():
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
    pc.create_index(
        name=index_name,
        dimension=dimensions,
        metric="cosine",
        spec=spec
    )
    
def get_index():
    return pc.Index(index_name)
