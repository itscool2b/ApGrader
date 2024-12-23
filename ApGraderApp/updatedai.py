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
import os
from dotenv import load_dotenv
from openai import OpenAI

# Import your Pinecone setup function
from .pineconesetup import get_index

###############################################################################
# 1) Load environment variables and set up your API keys
###############################################################################
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone index
index = get_index()

###############################################################################
# 2) get_relevant_documents: fetch from Pinecone
###############################################################################
def get_relevant_documents(query, prompt_type):
    """
    Retrieve relevant documents from Pinecone based on the query embedding 
    and the given prompt type. 
    Returns metadata with text, type, and grade if 'essay_type_grad_receivede' 
    starts with the correct prompt_type.
    """
    try:
        # Create the query embedding via OpenAI
        response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding

        # Query your Pinecone index
        results = index.query(
            vector=query_embedding,
            top_k=100,
            include_metadata=True
        )

        # Filter the results for docs whose 'essay_type_grad_receivede'
        # starts with the identified prompt_type
        filtered_results = []
        if "matches" in results:
            for match in results["matches"]:
                metadata = match.get("metadata", {})
                essay_metadata = metadata.get("essay_type_grad_receivede", "")
                # Check if it starts with the prompt_type
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
        raise RuntimeError(f"Error in embedding or querying Pinecone: {e}")


###############################################################################
# 3) Prompt Templates
###############################################################################

# ---- Classification Prompt ----
classification_prompt = PromptTemplate.from_template(
    """
Here is prompt for classification:

You are a teaching assistant for an AP U.S. History class. Your task is to read 
the LEQ prompt that a student has provided and determine which of the three main 
APUSH LEQ types it falls under:

- Comparison: The prompt asks the student to compare and/or contrast historical 
  developments, events, policies, or societies.
- Causation: The prompt asks the student to explain causes and/or effects of 
  historical events or developments.
- Continuity and Change Over Time (CCOT): The prompt asks the student to analyze 
  what changed and what remained the same over a particular time frame.

Instructions:
Read the provided LEQ prompt carefully.
Identify whether the prompt is a Comparison, Causation, or CCOT prompt.
Do not consider anything outside the prompt text itself—just classify it based 
on its wording and requirements.

Respond with only one of the three words: 
"Comparison" or "Causation" or "CCOT"

Student’s Prompt to Classify: {prompt}

The output should be exactly one word: "Comparison" or "Causation" or "CCOT"
"""
)

# ---- Evaluation Prompt ----
evaluation_prompt = PromptTemplate.from_template(
    """
You are an AP US History essay grader using the College Board's updated LEQ 
rubric from 2023. Your task is to evaluate a student's Long Essay Question (LEQ) 
strictly based on the rubric provided. All feedback, scores, and analysis must 
directly reference the rubric retrieved from the vector database. 
Do not introduce any criteria not explicitly included in the rubric.

The metadata structure for the essays you will use includes:
- `essay_type_grad_receivede`: e.g., "causation(3)", "comparison(6)", etc.
   - `prompt_type`: The type of the essay (e.g., "causation", "comparison", "ccot").
   - `grade`: The grade received (e.g., "3", "6").
- `text`: The essay content.

You must:
1. Ensure you filter essays matching the correct `prompt_type`.
2. Include all grades of that type (both "causation(6)" and "causation(3)").
3. Use the `text`, `prompt_type`, and `grade` as context while evaluating.

Scoring System:
Out of 6 points, following these categories:
- Contextualization (0–1 point)
- Thesis / Claim (0–1 point)
- Evidence (0–2 points)
- Analysis and Reasoning (0–2 points)

Use knowledge from AP US History textbooks (e.g., AMSCO) or other 
College Board-approved materials. 
Award no leniency: the essay only gets credit if it meets the rubric precisely.

Do not penalize minor grammar/spelling if the meaning is clear.

Integration of Sample Essays & Textbook:
Use the relevant sample essays (for {prompt_type}) from {relevant_docs} to 
benchmark. Fact-check historical claims using the same docs. 
Only award points if the essay meets the category requirements.

Now, you have the student's LEQ:

Prompt Type: {prompt_type}
Student Essay:
{student_essay}

Evaluate:
1. Does it have clear Contextualization?
2. Thesis/Claim out of 1 point?
3. Evidence (specific & used effectively) out of 2 points?
4. Analysis & Reasoning out of 2 points?

Finally:
- Provide a total score out of 6
- Give strict feedback referencing the rubric
"""
)

# ---- Optional Formatting Prompt ----
formatting_prompt = PromptTemplate.from_template(
    """
You are to reformat and structure the following evaluation text. 
Use headings for each rubric category, and ensure clarity.

Prompt Type: {prompt}
Evaluation Text:
{essay}

Rules:
- Separate categories with blank lines.
- Use bullet points or short paragraphs for clarity.
- If there's a total score, place it at the end, e.g.: "Total Score: X out of 6"
- Provide a succinct summary at the end labeled "Feedback Summary".
"""
)


###############################################################################
# 4) LLM Setup
###############################################################################
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o"
)

tools = [
    Tool(
        name="get rubric and sample essays",
        func=lambda query: "\n\n".join(get_relevant_documents(query)),
        description="Retrieve relevant rubric sections, example essays, or textbook chapters for the identified prompt type."
    )
]

from pydantic import BaseModel
from typing import List

###############################################################################
# 5) State Definition and Workflow
###############################################################################
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        prompt: The LEQ prompt from the student
        generation: (Optional) An LLM generation text
        documents: list of doc dictionaries
        prompt_type: string for classification (Comparison, Causation, CCOT)
        student_essay: the student's actual essay text
        evaluation: final evaluation text from LLM
        final: the formatted final string from the formatting node
    """
    prompt: str
    generation: str
    documents: List[dict]
    prompt_type: str
    student_essay: str
    evaluation: str
    final: str

workflow = StateGraph(GraphState)

def classify_prompt(state):
    """
    Node 1: Classify the student's LEQ prompt into Comparison, Causation, or CCOT.
    """
    response = llm.invoke(classification_prompt.format(prompt=state["prompt"]))
    state["prompt_type"] = response.content.strip()
    return state  # Return a dict

def retrieve_documents(state):
    """
    Node 2: Retrieve relevant docs (rubrics, sample essays, etc.) from Pinecone.
    """
    prompt_type = state["prompt_type"]
    query = (
        f"Retrieve rubric, example essays, and all relevant historical chapters "
        f"for {prompt_type} prompts to fact-check and provide feedback. "
        f"Ensure the documents cover the full context and key details."
    )
    docs = get_relevant_documents(query, prompt_type)
    state["documents"] = docs
    return state

def evaluate_essay(state):
    """
    Node 3: Use the LLM to evaluate the essay, referencing the retrieved docs 
    (which are in state["documents"]).
    """
    relevant_docs = "\n\n".join(doc.get("text", "") for doc in state["documents"])
    prompt_type = state["prompt_type"]
    student_essay = state["student_essay"]

    response = llm.invoke(
        evaluation_prompt.format(
            relevant_docs=relevant_docs,
            prompt_type=prompt_type,
            student_essay=student_essay
        )
    )
    # Save the LLM's raw evaluation text
    state["evaluation"] = response.content
    return state

def format_evaluation(state):
    """
    Node 4 (final): Optionally reformat the LLM evaluation text for readability.
    """
    raw_evaluation = state["evaluation"]
    prompt_type = state["prompt_type"]

    # Use the formatting prompt
    response = llm.invoke(
        formatting_prompt.format(
            prompt=prompt_type,
            essay=raw_evaluation
        )
    )
    state["final"] = response.content
    return state  # Return the updated dict

# Add nodes
workflow.add_node("classify_prompt", classify_prompt)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("evaluate_essay", evaluate_essay)
workflow.add_node("format_evaluation", format_evaluation)

# Define edges
workflow.add_edge(START, "classify_prompt")
workflow.add_edge("classify_prompt", "retrieve_documents")
workflow.add_edge("retrieve_documents", "evaluate_essay")
workflow.add_edge("evaluate_essay", "format_evaluation")
workflow.add_edge("format_evaluation", END)

# Compile the workflow
app = workflow.compile()

###############################################################################
# 6) The main evaluate() function (with flatten fix)
###############################################################################
def evaluate(prompt, essay):
    """
    Evaluate a student's essay based on the given prompt using the StateGraph workflow.
    The final node is 'format_evaluation', which stores the fully formatted text in state["final"].

    We'll flatten if needed, then return state["final"]. 
    If for some reason 'final' isn't present, we fallback to 'evaluation'.
    """
    try:
        initial_state = {
            "prompt": prompt,
            "generation": None,
            "documents": [],
            "prompt_type": None,
            "student_essay": essay,
            "evaluation": None,
            "final": None
        }

        evaluation_output = None
        # Step through the workflow
        for output in app.stream(initial_state):
            evaluation_output = output  # The final dictionary from last node

        # 1) Some versions of langgraph nest the final dict under the node name, e.g.:
        #    { "format_evaluation": { "final": "...text..." } }
        # Flatten if that's the case
        if evaluation_output and len(evaluation_output) == 1:
            only_key = list(evaluation_output.keys())[0]
            if only_key == "format_evaluation":
                evaluation_output = evaluation_output[only_key]

        # 2) If "final" is present, return it
        if evaluation_output and "final" in evaluation_output and evaluation_output["final"]:
            return evaluation_output["final"]

        # 3) Otherwise, fallback to "evaluation" if that's set
        if evaluation_output and "evaluation" in evaluation_output and evaluation_output["evaluation"]:
            return evaluation_output["evaluation"]

        # If still none, return error JSON
        return {
            "error": "No evaluation output generated",
            "details": "The workflow did not return a valid final or evaluation text."
        }

    except Exception as e:
        raise RuntimeError(f"Error during evaluation: {e}")
