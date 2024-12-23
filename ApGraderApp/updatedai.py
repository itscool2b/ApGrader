import os
from dotenv import load_dotenv
import openai
from langchain.agents import Tool, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from typing import List
from PyPDF2 import PdfReader
from typing_extensions import TypedDict
from io import BytesIO
from langgraph.graph import END, StateGraph, START
from .pineconesetup import get_index
import logging

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone index
index = get_index()


def get_relevant_documents(query, prompt_type):
    """
    Retrieve relevant documents from Pinecone based on the query embedding and prompt type.
    Includes all grades for the identified prompt type and returns metadata with text, type, and grade.
    """
    try:
        # Generate the query embedding
        response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = response["data"][0]["embedding"]

        # Query the vector database
        results = index.query(
            vector=query_embedding,
            top_k=100,
            include_metadata=True
        )

        # Filter documents based on the prompt type
        filtered_results = []
        for match in results["matches"]:
            metadata = match.get("metadata", {})
            essay_metadata = metadata.get("essay_type_grad_receivede", "")
            if essay_metadata.startswith(prompt_type):
                type_grade = essay_metadata.split("(")
                if len(type_grade) == 2:
                    grade = type_grade[1].rstrip(")")
                    filtered_results.append({
                        "text": metadata.get("text", ""),
                        "prompt_type": type_grade[0].strip(),
                        "grade": grade.strip()
                    })

        return filtered_results

    except Exception as e:
        logger.error(f"Error in embedding or querying Pinecone: {e}")
        raise RuntimeError(f"Error in embedding or querying Pinecone: {e}")


# Prompt templates
classification_prompt = PromptTemplate.from_template("""Here is prompt for classification:

You are a teaching assistant for an AP U.S. History class...
Student’s Prompt to Classify: {prompt}. The output should be one word "Comparison" "Causation" or "CCOT" """)

evaluation_prompt = PromptTemplate.from_template("""You are an AP US History essay grader...
Student Essay to Grade: {student_essay}

Evaluation Criteria...
""")


# Workflow tools
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o"
)

tools = [
    Tool(
        name="get rubric and sample essays",
        func=lambda query: "\n\n".join(get_relevant_documents(query)),
        description="Retrieve relevant sections of the rubric and example essays for grading."
    )
]


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The essay prompt
        generation: The LLM-generated evaluation
        documents: A list of relevant documents
        prompt_type: The type of the prompt
        student_essay: The essay content
    """
    question: str
    generation: str
    documents: List[str]
    prompt_type: str
    student_essay: str


# Define the workflow
workflow = StateGraph(GraphState)


def classify_prompt(state):
    question = state["question"]
    response = llm.predict(classification_prompt.format(prompt=question))
    state["prompt_type"] = response.strip()
    return state


def retrieve_documents(state):
    prompt_type = state["prompt_type"]
    query = f"Retrieve rubric, example essays, and all relevant historical chapters for {prompt_type} prompts."
    try:
        state["documents"] = get_relevant_documents(query, prompt_type)
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise RuntimeError(f"Error retrieving documents: {e}")
    return state


def evaluate_essay(state):
    student_essay = state["student_essay"]
    relevant_docs = "\n\n".join(state["documents"])
    prompt_type = state["prompt_type"]
    response = llm.predict(
        evaluation_prompt.format(
            relevant_docs=relevant_docs,
            prompt_type=prompt_type,
            student_essay=student_essay
        )
    )
    state["generation"] = response
    return state


workflow.add_node("classify_prompt", classify_prompt)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("evaluate_essay", evaluate_essay)

workflow.add_edge(START, "classify_prompt")
workflow.add_edge("classify_prompt", "retrieve_documents")
workflow.add_edge("retrieve_documents", "evaluate_essay")
workflow.add_edge("evaluate_essay", END)

app = workflow.compile()


def evaluate(prompt, essay):
    """
    Evaluate a student's essay based on the given prompt using the StateGraph workflow.
    Returns a structured evaluation with scores and feedback.
    """
    try:
        # Ensure initial_state contains all required keys
        initial_state = {
            "question": prompt or "",  # Default to an empty string if prompt is None
            "generation": None,
            "documents": [],
            "prompt_type": None,
            "student_essay": essay or ""  # Default to an empty string if essay is None
        }

        if not initial_state["question"].strip():
            raise ValueError("The 'prompt' is missing or empty.")
        if not initial_state["student_essay"].strip():
            raise ValueError("The 'student_essay' is missing or empty.")

        evaluation_output = None

        # Stream through the workflow outputs
        for output in app.stream(initial_state):
            evaluation_output = output

        # Ensure evaluation_output exists and contains the expected "generation" key
        if evaluation_output and "generation" in evaluation_output:
            return evaluation_output["generation"]

        logger.error("No evaluation output generated.")
        return {
            "error": "No evaluation output generated",
            "details": "The workflow did not return a valid evaluation."
        }

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise RuntimeError(f"Error during evaluation: {e}")
