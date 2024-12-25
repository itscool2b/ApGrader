##############################
# evaluate_ai.py
##############################

import os
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

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your environment.")

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")
index = get_index()

###############################################################################
# 2) get_relevant_documents: fetch from Pinecone
###############################################################################
def get_relevant_documents(query: str, prompt_type: str) -> List[Dict]:
    """
    Retrieve relevant documents from Pinecone based on:
    1) The query embedding (unless query is None, in which case we use a dummy vector).
    2) The prompt_type filter (if prompt_type is None, skip startswith check).

    Args:
        query (str): The search query for embeddings.
        prompt_type (str): The type of prompt to filter documents.

    Returns:
        List[Dict]: A list of filtered documents with 'text', 'prompt_type', and 'grade'.
    """
    try:
        if query is None:
            dummy_vector = [0.0] * 1536
            results = index.query(vector=dummy_vector, top_k=100, include_metadata=True)
        else:
            response = client.embeddings.create(input=query, model="text-embedding-ada-002")
            query_embedding = response.data[0].embedding
            results = index.query(vector=query_embedding, top_k=100, include_metadata=True)

        filtered_results = [
            {
                "text": match["metadata"].get("text", ""),
                "prompt_type": match["metadata"].get("essay_type_grad_received", "").split("(")[0].strip(),
                "grade": match["metadata"].get("essay_type_grad_received", "").split("(")[-1].strip(")")
            }
            for match in results.get("matches", [])
        ]
        return filtered_results
    except Exception as e:
        logging.error(f"Error querying Pinecone: {e}")
        raise RuntimeError("Error querying Pinecone.") from e

###############################################################################
# 3) Define prompt templates
###############################################################################
classification_prompt = PromptTemplate.from_template(
    """
Classify the LEQ prompt into one of these types: Comparison, Causation, or CCOT. 
Respond with only one word.

Prompt: {prompt}
"""
)

thesis_prompt = PromptTemplate.from_template(
    """
Evaluate the thesis statement based on the rubric:
- Responds to the prompt with a defensible thesis.
- Establishes a line of reasoning.

Essay: {essay}
Prompt Type: {prompt_type}

Output:
- Score (0 or 1)
- Feedback
"""
)

contextualization_prompt = PromptTemplate.from_template(
    """
Evaluate the contextualization in the essay based on the rubric:
- Describes a broader historical context relevant to the prompt.

Essay: {essay}
Prompt Type: {prompt_type}

Output:
- Score (0 or 1)
- Feedback
"""
)

evidence_prompt = PromptTemplate.from_template(
    """
Evaluate evidence usage in the essay:
1. Identifies at least two relevant historical examples (1 point).
2. Uses evidence to support argument (1 point).

Essay: {essay}
Prompt Type: {prompt_type}

Output:
- Score (0, 1, or 2)
- Feedback
"""
)

complexunderstanding_prompt = PromptTemplate.from_template(
    """
Evaluate analysis and reasoning:
1. Historical Reasoning (1 point).
2. Complex Understanding (1 point).

Essay: {essay}
Prompt Type: {prompt_type}

Output:
- Score (0, 1, or 2)
- Feedback
"""
)

summation_prompt = PromptTemplate.from_template(
    """
Summarize the evaluation:
Thesis: {thesis_generation}
Contextualization: {contextualization_generation}
Evidence: {evidence_generation}
Analysis and Reasoning: {complexunderstanding_generation}

Output:
- Total Score (0-6)
- Strengths
- Weaknesses
- Suggestions for improvement
"""
)

###############################################################################
# 4) Define workflow and states
###############################################################################
class GraphState(TypedDict):
    prompt: str
    prompt_type: str
    student_essay: str
    thesis_generation: str
    contextualization_generation: str
    evidence_generation: str
    complexunderstanding_generation: str
    summation: str

workflow = StateGraph(GraphState)

###############################################################################
# 5) Workflow node definitions
###############################################################################
def classify_prompt_node(state: GraphState) -> GraphState:
    prompt = state.get("prompt", "")
    response = llm.invoke(classification_prompt.format(prompt=prompt)).content.strip()
    state["prompt_type"] = response
    return state

def thesis_grading_node(state: GraphState) -> GraphState:
    essay = state.get("student_essay", "")
    prompt_type = state.get("prompt_type", "")
    response = llm.invoke(thesis_prompt.format(essay=essay, prompt_type=prompt_type)).content.strip()
    state["thesis_generation"] = response
    return state

def contextualization_grading_node(state: GraphState) -> GraphState:
    essay = state.get("student_essay", "")
    prompt_type = state.get("prompt_type", "")
    response = llm.invoke(contextualization_prompt.format(essay=essay, prompt_type=prompt_type)).content.strip()
    state["contextualization_generation"] = response
    return state

def evidence_grading_node(state: GraphState) -> GraphState:
    essay = state.get("student_essay", "")
    prompt_type = state.get("prompt_type", "")
    response = llm.invoke(evidence_prompt.format(essay=essay, prompt_type=prompt_type)).content.strip()
    state["evidence_generation"] = response
    return state

def analysis_grading_node(state: GraphState) -> GraphState:
    essay = state.get("student_essay", "")
    prompt_type = state.get("prompt_type", "")
    response = llm.invoke(complexunderstanding_prompt.format(essay=essay, prompt_type=prompt_type)).content.strip()
    state["complexunderstanding_generation"] = response
    return state

def final_node(state: GraphState) -> GraphState:
    """
    Final node to generate the summation from all partial sections.
    """
    try:
        logging.info("Generating final summation.")

        # Extract required inputs
        thesis = state.get("thesis_generation", "")
        contextualization = state.get("contextualization_generation", "")
        evidence = state.get("evidence_generation", "")
        complex_understanding = state.get("complexunderstanding_generation", "")

        # Log the inputs for debugging
        logging.debug(f"Inputs for final summation - Thesis: {thesis}, Contextualization: {contextualization}, Evidence: {evidence}, Complex Understanding: {complex_understanding}")

        # Check for missing inputs
        if not all([thesis, contextualization, evidence, complex_understanding]):
            logging.warning("One or more inputs for summation are missing.")
            state["summation"] = None
            return state

        # Format the prompt
        formatted_prompt = summation_prompt.format(
            thesis_generation=thesis,
            contextualization_generation=contextualization,
            evidence_generation=evidence,
            complexunderstanding_generation=complex_understanding,
        )
        logging.debug(f"Summation Prompt: {formatted_prompt}")

        # Generate response
        response = llm.invoke(formatted_prompt).content.strip()
        logging.debug(f"Summation Response: {response}")

        state["summation"] = response
        return state

    except Exception as e:
        logging.error(f"Error in final_node: {e}")
        state["summation"] = None
        return state


###############################################################################
# 6) Workflow construction
###############################################################################
workflow.add_node("classify_prompt", classify_prompt_node)
workflow.add_node("thesis_grading", thesis_grading_node)
workflow.add_node("contextualization_grading", contextualization_grading_node)
workflow.add_node("evidence_grading", evidence_grading_node)
workflow.add_node("analysis_grading", analysis_grading_node)
workflow.add_node("final_node", final_node)

workflow.add_edge(START, "classify_prompt")
workflow.add_edge("classify_prompt", "thesis_grading")
workflow.add_edge("thesis_grading", "contextualization_grading")
workflow.add_edge("contextualization_grading", "evidence_grading")
workflow.add_edge("evidence_grading", "analysis_grading")
workflow.add_edge("analysis_grading", "final_node")
workflow.add_edge("final_node", END)

app = workflow.compile()

###############################################################################
# 7) Evaluate function
###############################################################################
def evaluate(prompt: str, essay: str) -> Dict:
    """
    Evaluate a student's essay using the StateGraph workflow.
    """
    try:
        initial_state: GraphState = {
            "prompt": prompt,
            "prompt_type": "",
            "student_essay": essay,
            "thesis_generation": "",
            "contextualization_generation": "",
            "evidence_generation": "",
            "complexunderstanding_generation": "",
            "summation": "",
        }

        for output in app.stream(initial_state):
            logging.info(f"Intermediate state: {output}")
            final_output = output

        if final_output.get("summation"):
            logging.info("Summation successfully generated.")
            return {"status": "success", "result": final_output["summation"]}

        logging.warning("Summation not generated.")
        return {"status": "error", "message": "Summation not generated.", "details": final_output}

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        return {"status": "error", "message": str(e)}
