##############################
# evaluate_ai.py
##############################

import os
import openai
import json
import logging
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from ApGraderApp.p import setup_index, get_index

from typing import List, Dict
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START

###############################################################################
# 1) Load environment variables and set up your API keys
###############################################################################
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Verify that the OpenAI API key is loaded
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your environment.")

# Set OpenAI API key for the openai library
openai.api_key = OPENAI_API_KEY
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone index
index = get_index()

###############################################################################
# 2) get_relevant_documents: fetch from Pinecone
###############################################################################
def get_relevant_documents(query: str, prompt_type: str) -> List[Dict]:
    """
    Retrieve relevant documents from Pinecone based on:
    1) The query embedding (unless query is None, in which case we do a 'dummy' query).
    2) The prompt_type filter (unless prompt_type is None, then we skip startswith check).

    Args:
        query (str): The search query for embeddings.
        prompt_type (str): The type of prompt to filter documents.

    Returns:
        List[Dict]: A list of filtered documents with 'text', 'prompt_type', and 'grade'.
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
            response = openai.Embedding.create(
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
        if "matches" in results:
            for match in results["matches"]:
                metadata = match.get("metadata", {})
                essay_metadata = metadata.get("essay_type_grad_received", "")

                if prompt_type is None or essay_metadata.lower().startswith(prompt_type.lower()):
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
        logging.error(f"Error in embedding or querying Pinecone: {e}")
        raise RuntimeError("Error in embedding or querying Pinecone.") from e

###############################################################################
# 3) Prompt Templates
###############################################################################
classification_prompt = PromptTemplate.from_template(
    """
You are a highly accurate and strict teaching assistant for an AP U.S. History class. Your task is to read the LEQ prompt provided by a student and determine which of the three main APUSH LEQ types it falls under:
- **Comparison**: The prompt asks the student to compare and/or contrast historical developments, events, policies, or societies.
- **Causation**: The prompt asks the student to explain causes and/or effects of historical events or developments.
- **Continuity and Change Over Time (CCOT)**: The prompt asks the student to analyze what changed and what remained the same over a particular time frame.

**Instructions**:
1. Read the provided LEQ prompt carefully.
2. Identify whether the prompt is a **Comparison**, **Causation**, or **CCOT** prompt.
3. **Respond with only one of the three exact words**: "Comparison", "Causation", or "CCOT". **Do not include any additional text, explanations, or characters.**

**Student’s Prompt to Classify**: {prompt}

**Your Response**:
"""
)

thesis_prompt = PromptTemplate.from_template(
    """
Evaluate the thesis statement in the following essay based on the provided rubric and evaluation standards:

**Rubric for Thesis / Claim**:
- Responds to the prompt with a historically defensible thesis or claim.
- Establishes a line of reasoning.
- Makes a claim that responds to the prompt (not merely restating or rephrasing it).
- Consists of one or more sentences located in one place, either in the introduction or conclusion.

**Essay to Evaluate**:
{essay}

**Prompt Type**: {prompt_type}

**Output**:
- **Score (0 or 1)**: Indicate whether the thesis earns the point.
- **Feedback**: Provide a brief explanation justifying the score.
"""
)

contextualization_prompt = PromptTemplate.from_template(
    """
Evaluate the contextualization in the following essay based on the provided rubric and evaluation standards:

**Rubric for Contextualization**:
- Describes a broader historical context relevant to the prompt.
- Relates the topic to broader historical events, developments, or processes before, during, or after the time frame of the question.
- Not awarded for merely a phrase or reference.

**Essay to Evaluate**:
{essay}

**Prompt Type**: {prompt_type}

**Output**:
- **Score (0 or 1)**: Indicate whether the contextualization earns the point.
- **Feedback**: Provide a brief explanation justifying the score.
"""
)

evidence_prompt = PromptTemplate.from_template(
    """
Evaluate the evidence in the following essay based on the provided rubric and evaluation standards:

**Rubric for Evidence**:
1. **Specific Evidence Usage (1 point)**: At least two specific, relevant historical examples.
2. **Evidence Supporting Argument (1 point)**: Evidence must be integrated to support the thesis.

**Essay to Evaluate**:
{essay}

**Prompt Type**: {prompt_type}

**Output**:
- **Score (0, 1, or 2)**: Indicate the total points for Evidence.
- **Feedback**: Provide a brief explanation justifying the score.
"""
)

complexunderstanding_prompt = PromptTemplate.from_template(
    """
Evaluate the analysis and reasoning in the following essay based on the provided rubric and evaluation standards:

**Rubric for Analysis and Reasoning**:
1. **Historical Reasoning (1 point)**: Uses reasoning such as causation, comparison, or CCOT.
2. **Complex Understanding (1 point)**: Demonstrates nuanced understanding or connections.

**Essay to Evaluate**:
{essay}

**Prompt Type**: {prompt_type}

**Output**:
- **Score (0, 1, or 2)**: Indicate the total points for Analysis and Reasoning.
- **Feedback**: Provide a brief explanation justifying the score.
"""
)

summation_prompt = PromptTemplate.from_template(
    """
Based on the feedback and scores provided for each section of the essay, generate a comprehensive evaluation:

**Scores**:
- Thesis: {thesis_generation}
- Contextualization: {contextualization_generation}
- Evidence: {evidence_generation}
- Analysis and Reasoning: {complexunderstanding_generation}

**Total Score (0-6)**: Calculate the total score.

**Summary**:
Provide a summary of strengths, weaknesses, and areas for improvement.
"""
)

###############################################################################
# 4) Workflow Definition
###############################################################################

class GraphState(TypedDict):
    """
    Represents the state of the graph workflow.
    """
    prompt: str
    prompt_type: str
    student_essay: str
    rubric: List[Dict]
    thesis_generation: str
    contextualization_generation: str
    evidence_generation: str
    complexunderstanding_generation: str
    summation: str

workflow = StateGraph(GraphState)

###############################################################################
# 5) Node Functions
###############################################################################

def classify_prompt_node(state: GraphState) -> GraphState:
    prompt = state.get("prompt", "").strip()
    formatted_prompt = classification_prompt.format(prompt=prompt)
    response = llm.invoke(formatted_prompt).content.strip()
    state["prompt_type"] = response
    return state

def fetch_rubric_node(state: GraphState) -> GraphState:
    state["rubric"] = get_relevant_documents("LEQ Rubric", None)
    return state

def retrieve_essays_node(state: GraphState) -> GraphState:
    essays = get_relevant_documents(None, state.get("prompt_type"))
    state["documents"] = essays
    return state

def thesis_grading_node(state: GraphState) -> GraphState:
    formatted_prompt = thesis_prompt.format(
        essay=state.get("student_essay"),
        prompt_type=state.get("prompt_type")
    )
    response = llm.invoke(formatted_prompt).content.strip()
    state["thesis_generation"] = response
    return state

def final_node(state: GraphState) -> GraphState:
    formatted_prompt = summation_prompt.format(
        thesis_generation=state.get("thesis_generation"),
        contextualization_generation=state.get("contextualization_generation"),
        evidence_generation=state.get("evidence_generation"),
        complexunderstanding_generation=state.get("complexunderstanding_generation")
    )
    state["summation"] = llm.invoke(formatted_prompt).content.strip()
    return state

workflow.add_node("classify_prompt", classify_prompt_node)
workflow.add_node("fetch_rubric", fetch_rubric_node)
workflow.add_node("retrieve_essays", retrieve_essays_node)
workflow.add_node("thesis_grading", thesis_grading_node)
workflow.add_node("final_node", final_node)

workflow.add_edge(START, "classify_prompt")
workflow.add_edge("classify_prompt", "fetch_rubric")
workflow.add_edge("fetch_rubric", "retrieve_essays")
workflow.add_edge("retrieve_essays", "thesis_grading")
workflow.add_edge("thesis_grading", "final_node")
workflow.add_edge("final_node", END)
app = workflow.compile()

###############################################################################
# 6) Main Function
###############################################################################

def evaluate(prompt: str, essay: str) -> Dict:
    initial_state: GraphState = {
        "prompt": prompt,
        "prompt_type": None,
        "student_essay": essay,
        "rubric": [],
        "thesis_generation": None,
        "contextualization_generation": None,
        "evidence_generation": None,
        "complexunderstanding_generation": None,
        "summation": None,
    }
    final_output = app.run(initial_state)
    return final_output.get("summation", "Evaluation could not be completed.")
