import os
import openai
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

import ApGraderApp.p as p
from ApGraderApp.p import pc, setup_index, get_index

from typing import List, Dict
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START

OPENAI_API_KEY = os.getenv('OPENAI_APi_KEY')

openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

index = get_index()

def retriever(query: str, top_k: int = 100) -> List[Dict]:
    """
    Generalized function to retrieve relevant documents from Pinecone based on a query.

    Args:
        query (str): The search query.
        top_k (int): Number of top results to retrieve. Default is 100.

    Returns:
        List[Dict]: A list of retrieved documents with 'text' and 'metadata'.
    """
    try:
       
        response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding

        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        return [
            {
                "text": match.get("metadata", {}).get("text", ""),
                "metadata": match.get("metadata", {})
            }
            for match in results.get("matches", [])
        ]
    except Exception as e:
        logging.error(f"Error in retriever: {e}")
        raise RuntimeError("Error in retriever function.") from e
    
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")