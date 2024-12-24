import os
from dotenv import load_dotenv
import openai

from langchain.agents import Tool, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI as LegacyChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings as LegacyOpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from typing import List
from PyPDF2 import PdfReader
from typing_extensions import TypedDict
from io import BytesIO

from langgraph.graph import END, StateGraph, START

# Import your Pinecone setup function
from .pineconesetup import get_index

###############################################################################
# 1) Load environment variables and set up your API keys
###############################################################################
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai

# Initialize Pinecone index
index = get_index()

###############################################################################
# 2) get_relevant_documents: fetch from Pinecone
###############################################################################
def get_relevant_documents(query, prompt_type):
    """
    Retrieve relevant documents from Pinecone based on:
    1) The query embedding (unless query is None, in which case we do a 'dummy' query).
    2) The prompt_type filter (unless prompt_type is None, then we skip startswith check).
    """
    try:
        if query is None:
            dummy_vector = [0.0] * 1536
            results = index.query(
                vector=dummy_vector,
                top_k=100,
                include_metadata=True
            )
        else:
            response = client.Embedding.create(
                input=query,
                model="text-embedding-ada-002"
            )
            query_embedding = response["data"][0]["embedding"]
            results = index.query(
                vector=query_embedding,
                top_k=100,
                include_metadata=True
            )

        filtered_results = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            essay_metadata = metadata.get("essay_type_grad_receivede", "")
            if prompt_type is None or essay_metadata.startswith(prompt_type):
                type_grade = essay_metadata.split("(")
                grade = type_grade[1].rstrip(")") if len(type_grade) == 2 else ""
                filtered_results.append({
                    "text": metadata.get("text", ""),
                    "prompt_type": type_grade[0].strip() if type_grade else "",
                    "grade": grade
                })

        return filtered_results

    except Exception as e:
        raise RuntimeError(f"Error querying Pinecone: {e}")


###############################################################################
# 3) Prompt Templates
###############################################################################
classification_prompt = PromptTemplate.from_template(
    """
You are a teaching assistant for an AP U.S. History class. Your task is to classify the provided LEQ prompt as one of the following:
- "Comparison"
- "Causation"
- "Continuity and Change Over Time (CCOT)"
Prompt: {prompt}
Respond with only one of the words: "Comparison", "Causation", or "CCOT".
"""
)

evaluation_prompt = PromptTemplate.from_template(
    """
Evaluate the following essay based on the APUSH rubric for prompt type {prompt_type}.
Rubric Details: {rubric}
Student Essay: {student_essay}
Provide a score and feedback for each rubric criterion.
"""
)

thesis_prompt = PromptTemplate.from_template(
    """
Evaluate the thesis in the essay below using the rubric:
Rubric: {rubric}
Essay: {essay}
Type: {prompt_type}
Provide a score (0-1) and detailed feedback.
"""
)

contextualization_prompt = PromptTemplate.from_template(
    """
Evaluate contextualization in the essay below using the rubric:
Rubric: {rubric}
Essay: {essay}
Type: {prompt_type}
Provide a score (0-1) and detailed feedback.
"""
)

evidence_prompt = PromptTemplate.from_template(
    """
Evaluate evidence usage in the essay below using the rubric:
Rubric: {rubric}
Essay: {essay}
Type: {prompt_type}
Provide a score (0-2) and detailed feedback.
"""
)

complexunderstanding_prompt = PromptTemplate.from_template(
    """
Evaluate analysis and reasoning in the essay below using the rubric:
Rubric: {rubric}
Essay: {essay}
Type: {prompt_type}
Provide a score (0-2) and detailed feedback.
"""
)

summation_prompt = PromptTemplate.from_template(
    """
Summarize the following evaluation sections:
- Thesis: {thesis_generation}
- Contextualization: {contextualization_generation}
- Evidence: {evidence_generation}
- Analysis: {complexunderstanding_generation}
Provide a total score (0-6) and overall feedback.
"""
)


###############################################################################
# 4) LLM Setup
###############################################################################
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o")


###############################################################################
# 5) State Definition and Workflow
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


###############################################################################
# Workflow Nodes
###############################################################################
def classify_prompt_node(state):
    response = llm.invoke(classification_prompt.format(prompt=state["prompt"]))
    state["prompt_type"] = response.content.strip()
    return state


def fetch_rubric_node(state):
    state["rubric"] = get_relevant_documents("LEQ Rubric", None)
    return state


def thesis_grading_node(state):
    response = llm.invoke(
        thesis_prompt.format(
            rubric=state["rubric"],
            essay=state["student_essay"],
            prompt_type=state["prompt_type"]
        )
    )
    state["thesis_generation"] = response.content
    return state


def contextualization_grading_node(state):
    response = llm.invoke(
        contextualization_prompt.format(
            rubric=state["rubric"],
            essay=state["student_essay"],
            prompt_type=state["prompt_type"]
        )
    )
    state["contextualization_generation"] = response.content
    return state


def evidence_grading_node(state):
    response = llm.invoke(
        evidence_prompt.format(
            rubric=state["rubric"],
            essay=state["student_essay"],
            prompt_type=state["prompt_type"]
        )
    )
    state["evidence_generation"] = response.content
    return state


def analysis_grading_node(state):
    response = llm.invoke(
        complexunderstanding_prompt.format(
            rubric=state["rubric"],
            essay=state["student_essay"],
            prompt_type=state["prompt_type"]
        )
    )
    state["complexunderstanding_generation"] = response.content
    return state


def final_node(state):
    response = llm.invoke(
        summation_prompt.format(
            thesis_generation=state["thesis_generation"],
            contextualization_generation=state["contextualization_generation"],
            evidence_generation=state["evidence_generation"],
            complexunderstanding_generation=state["complexunderstanding_generation"]
        )
    )
    state["summation"] = response.content
    return state


###############################################################################
# Workflow Setup
###############################################################################
workflow.add_node("classify_prompt", classify_prompt_node)
workflow.add_node("fetch_rubric", fetch_rubric_node)
workflow.add_node("thesis_grading", thesis_grading_node)
workflow.add_node("contextualization_grading", contextualization_grading_node)
workflow.add_node("evidence_grading", evidence_grading_node)
workflow.add_node("analysis_grading", analysis_grading_node)
workflow.add_node("final_node", final_node)

workflow.add_edge(START, "classify_prompt")
workflow.add_edge("classify_prompt", "fetch_rubric")
workflow.add_edge("fetch_rubric", "thesis_grading")
workflow.add_edge("thesis_grading", "contextualization_grading")
workflow.add_edge("contextualization_grading", "evidence_grading")
workflow.add_edge("evidence_grading", "analysis_grading")
workflow.add_edge("analysis_grading", "final_node")
workflow.add_edge("final_node", END)

app = workflow.compile()


###############################################################################
# Main Function
###############################################################################
def evaluate(prompt, essay):
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

    try:
        for output in app.stream(initial_state):
            if output.get("summation"):
                return output["summation"]
        return {"error": "No evaluation output generated"}
    except Exception as e:
        raise RuntimeError(f"Error during evaluation: {e}")