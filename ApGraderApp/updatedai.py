import os
from dotenv import load_dotenv
import openai
import logging

from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI as LegacyChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings as LegacyOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from fastapi import FastAPI, Request
from openai import OpenAI
from .pineconesetup import get_index

###############################################################################
# Setup Logging
###############################################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# Load environment variables and API keys
###############################################################################
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in the environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)
index = get_index()  # Load the Pinecone index

###############################################################################
# Helper Function: get_relevant_documents
###############################################################################
def get_relevant_documents(query, prompt_type):
    try:
        if query is None:
            dummy_vector = [0.0] * 1536
            results = index.query(vector=dummy_vector, top_k=100, include_metadata=True)
        else:
            response = client.embeddings.create(input=query, model="text-embedding-ada-002")
            query_embedding = response.data[0].embedding
            results = index.query(vector=query_embedding, top_k=100, include_metadata=True)

        filtered_results = []
        if "matches" in results:
            for match in results["matches"]:
                metadata = match.get("metadata", {})
                essay_metadata = metadata.get("essay_type_grad_receivede", "")
                if prompt_type is None or essay_metadata.startswith(prompt_type):
                    type_grade = essay_metadata.split("(")
                    if len(type_grade) == 2:
                        grade = type_grade[1].rstrip(")")
                        filtered_results.append({
                            "text": metadata.get("text", ""),
                            "prompt_type": type_grade[0].strip(),
                            "grade": grade.strip()
                        })
                    else:
                        filtered_results.append({
                            "text": metadata.get("text", ""),
                            "prompt_type": essay_metadata.strip(),
                            "grade": ""
                        })
        return filtered_results
    except Exception as e:
        logger.error(f"Error in embedding or querying Pinecone: {e}")
        raise RuntimeError(f"Error in embedding or querying Pinecone: {e}")

###############################################################################
# Prompt Templates
###############################################################################
classification_prompt = PromptTemplate.from_template("""
You are a teaching assistant for an AP U.S. History class. Your task is to classify this LEQ prompt:
{prompt}
Respond with "Comparison", "Causation", or "CCOT".
""")

evaluation_prompt = PromptTemplate.from_template("""
Evaluate this essay based on the rubric:
{rubric}
Student Essay:
{student_essay}
Prompt Type: {prompt_type}
""")

###############################################################################
# LLM Setup
###############################################################################
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4")

###############################################################################
# StateGraph Workflow Setup
###############################################################################
class GraphState(TypedDict):
    prompt: str
    prompt_type: str
    student_essay: str
    rubric: List[dict]
    thesis_generation: str
    contextualization_generation: str
    evidence_generation: str
    complexunderstanding_generation: str
    summation: str

workflow = StateGraph(GraphState)

def classify_prompt_node(state):
    response = llm.invoke(classification_prompt.format(prompt=state["prompt"]))
    state["prompt_type"] = response.content.strip()
    return state

def fetch_rubric_node(state):
    docs = get_relevant_documents("LEQ Rubric", None)
    state["rubric"] = docs
    return state

def thesis_grading_node(state):
    rubric = state["rubric"]
    essay = state["student_essay"]
    prompt_type = state["prompt_type"]
    response = llm.invoke(evaluation_prompt.format(rubric=rubric, student_essay=essay, prompt_type=prompt_type))
    state["thesis_generation"] = response.content
    return state

def final_node(state):
    response = llm.invoke(evaluation_prompt.format(
        rubric=state["rubric"],
        student_essay=state["student_essay"],
        prompt_type=state["prompt_type"]
    ))
    state["summation"] = response.content
    return state

workflow.add_node("classify_prompt", classify_prompt_node)
workflow.add_node("fetch_rubric", fetch_rubric_node)
workflow.add_node("thesis_grading", thesis_grading_node)
workflow.add_node("final_node", final_node)

workflow.add_edge(START, "classify_prompt")
workflow.add_edge("classify_prompt", "fetch_rubric")
workflow.add_edge("fetch_rubric", "thesis_grading")
workflow.add_edge("thesis_grading", "final_node")
workflow.add_edge("final_node", END)

app = workflow.compile()

###############################################################################
# FastAPI Application
###############################################################################
api = FastAPI()

@api.post("/ApGrader/process")
async def process(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        essay = data.get("essay", "")

        if not prompt or not essay:
            return {"error": "Missing 'prompt' or 'essay' in the request."}

        initial_state = {
            "prompt": prompt,
            "prompt_type": None,
            "student_essay": essay,
            "rubric": [],
            "thesis_generation": None,
            "contextualization_generation": None,
            "evidence_generation": None,
            "complexunderstanding_generation": None,
            "summation": None
        }

        final_output = None
        for output in app.stream(initial_state):
            final_output = output

        if final_output and "summation" in final_output:
            return {"evaluation": final_output["summation"]}

        return {"error": "Evaluation failed", "details": "No valid summation generated."}
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return {"error": "Internal Server Error", "details": str(e)}