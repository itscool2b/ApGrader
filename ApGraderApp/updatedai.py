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
    Retrieve relevant documents from Pinecone based on the query embedding and prompt type.
    Includes all grades for the identified prompt type and returns metadata with text, type, and grade.
    """
    try:
        response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding

        results = index.query(
            vector=query_embedding,
            top_k=100,
            include_metadata=True
        )

        filtered_results = []
        if "matches" in results:
            for match in results["matches"]:
                metadata = match.get("metadata", {})
                essay_metadata = metadata.get("essay_type_grad_receivede", "")
                # Only keep docs whose 'essay_type_grad_receivede' starts with prompt_type
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

You are a teaching assistant for an AP U.S. History class. Your task is to read the LEQ prompt that a student has provided and determine which of the three main APUSH LEQ types it falls under:
Comparison: The prompt asks the student to compare and/or contrast historical developments, events, policies, or societies.
Causation: The prompt asks the student to explain causes and/or effects of historical events or developments.
Continuity and Change Over Time (CCOT): The prompt asks the student to analyze what changed and what remained the same over a particular time frame.

Instructions:
Read the provided LEQ prompt carefully.
Identify whether the prompt is a Comparison, Causation, or CCOT prompt.
Do not consider anything outside the prompt text itself—just classify it based on its wording and requirements.
Respond with only one of the three words: "Comparison" "Causation" or "CCOT" depending on which category best matches the prompt.

Student’s Prompt to Classify: {prompt}
The output should be one word "Comparison" "Causation" or "CCOT"
"""
)

# ---- Evaluation Prompt ----
evaluation_prompt = PromptTemplate.from_template(
    """
You are an AP US History essay grader using the College Board's updated LEQ rubric from 2023. 
Your task is to evaluate a student's Long Essay Question (LEQ) strictly based on the rubric provided. 
All feedback, scores, and analysis must directly reference the rubric retrieved from the vector database. 
Do not introduce any criteria not explicitly included in the rubric.

The metadata structure for the essays you will use includes:
- `essay_type_grad_receivede`: A string in the format `prompt_type(grade)` (e.g., "causation(3)").
   - `prompt_type`: The type of the essay (e.g., "causation", "comparison", or "ccot").
   - `grade`: The grade received for the essay (e.g., "3", "6").
- `text`: The essay content.

You must take the entire structure into account:
1. Ensure you filter essays matching the correct `prompt_type`.
2. Include all grades of the identified type (e.g., both "causation(6)" and "causation(3)").
3. Use the `text`, `prompt_type`, and `grade` as context while evaluating.

The scoring system is out of 6 points, and your grading must align with realistic standards used by AP US History graders. Apply the same standards that average AP graders have, ensuring that all points awarded demonstrate strong adherence to the rubric. Use knowledge from actual AP US History textbooks, such as AMSCO, or other College Board-approved materials. Be strict and apply no leniency in your grading. The essay should only receive credit when it meets the criteria outlined in the rubric. If the essay does not fulfill the requirements for a point, do not award partial credit or attempt to rationalize its inclusion.

Do not take spelling and grammar into account when evaluating the essay. Focus solely on the content and how it aligns with the rubric criteria. As long as the meaning of the essay is clear, spelling and grammar errors should not impact the evaluation.

Ensure you pay attention to this prompt as it directly dictates the context and focus of your evaluation. Do not overemphasize the prompt, but make sure it is clearly understood and integrated into your grading process.

Integration of Sample Essays and Textbook Chapters:
- Sample Essays: Utilize the sample essays provided for the specific prompt type ({prompt_type}) as benchmarks.
  These samples exemplify the range of scores from 0 to 6 points.
  While using these examples, recognize that there are multiple valid approaches to achieving high scores.
  Do not penalize a student for deviating in content or structure from the sample essays as long as their essay meets the rubric criteria effectively.
- Textbook Chapters: Reference the relevant chapters from AP US History textbooks included in {relevant_docs} to fact-check historical information presented in the student's essay.
  Ensure that all factual claims, dates, names, and events are accurate and align with the approved textbook content.

Evaluation Process
First Read:
- Determine the Thesis/Claim: Identify the thesis, often located in the introduction or conclusion.
  Determine:
    - Is the thesis clear, historically defensible, and does it answer the prompt?
    - Does it establish a clear line of reasoning?
- Scan for Contextualization: Assess whether the essay includes a broader historical context for the prompt. Look for:
    - A detailed explanation of events, trends, or processes occurring before, during, or after the prompt’s timeframe.
    - Whether the context is relevant and helps set up the thesis.

Second Read:
- Check for Evidence:
  - Specific Evidence: Does the student use specific and accurate historical examples (e.g., dates, names, events)?
  - Evidence Supporting Argument: Do the examples connect logically to the argument? Are they used to support the thesis effectively?
- Evaluate Reasoning and Analysis:
  - Historical Reasoning: Does the essay use historical reasoning skills (e.g., causation, comparison, continuity and change)?
  - Complexity: Does the student demonstrate a nuanced understanding of history, such as showing multiple perspectives, considering counter arguments, or synthesizing ideas across time periods?

The rubric, textbook chapters, and sample essays of the same prompt type are given below. Take these into consideration while grading:
{relevant_docs}

This is the prompt type for the essay which you should take into consideration while grading:
{prompt_type}

Student Essay to Grade:
{student_essay}

Evaluation Criteria

Contextualization (0–1 point):
- 1 point: Awarded only if the essay describes a broader historical context relevant to the prompt. 
- 0 points: Do not award points for vague or unrelated context.

Thesis / Claim (0–1 point):
- 1 point: Awarded if the student responds to the prompt with a historically defensible thesis that establishes a line of reasoning.
- 0 points: Do not award points for restatement of the prompt, overgeneralized statements, or claims lacking a line of reasoning.

Evidence (0–2 points):
- Specific Evidence (1 point): At least two specific, relevant historical examples. 
- Evidence Supporting Argument (1 point): The essay must use at least two pieces of evidence effectively to support the argument.

Analysis and Reasoning (0–2 points):
- Historical Reasoning (1 point): The response demonstrates at least one historical reasoning skill (comparison, causation, CCOT) to structure the argument.
- Complex Understanding (1 point): The essay must demonstrate a complex understanding of the historical development.

Output Format:
- Contextualization (0-1 point): [Score w/ feedback]
- Thesis / Claim (0-1 point): [Score w/ feedback]
- Evidence (0-2 points):
  - Specific Evidence: [Score w/ feedback]
  - Evidence Supporting Argument: [Score w/ feedback]
- Analysis and Reasoning (0-2 points):
  - Historical Reasoning: [Score w/ feedback]
  - Complex Understanding: [Score w/ feedback]
- Total Score (out of 6): [Score]

Feedback Summary:
Provide a strict summary of strengths, weaknesses, and areas for improvement. Draw on approved AP US History materials. The total score is out of 6 points. Award no partial or “benefit-of-the-doubt” credit if the criteria is not explicitly met.

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
        description="Retrieve relevant sections of the rubric and example essays for grading. Use the entire thing."
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
        prompt_type: str for classification (Comparison, Causation, CCOT)
        student_essay: the student's actual essay text
        evaluation: final evaluation text from LLM
    """
    prompt: str
    generation: str
    documents: List[dict]
    prompt_type: str
    student_essay: str
    evaluation: str

workflow = StateGraph(GraphState)

def classify_prompt(state):
    """
    Node 1: Classify the student's LEQ prompt.
    """
    response = llm.invoke(classification_prompt.format(prompt=state["prompt"]))
    state["prompt_type"] = response.content.strip()
    return state  # Must return a dict

def retrieve_documents(state):
    """
    Node 2: Retrieve relevant docs from Pinecone based on 'prompt_type'.
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
    Node 3: Use the LLM to evaluate the essay, referencing the retrieved docs.
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
    # Put the final feedback text into state["evaluation"]
    state["evaluation"] = response.content
    return state

workflow.add_node("classify_prompt", classify_prompt)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("evaluate_essay", evaluate_essay)

workflow.add_edge(START, "classify_prompt")
workflow.add_edge("classify_prompt", "retrieve_documents")
workflow.add_edge("retrieve_documents", "evaluate_essay")
workflow.add_edge("evaluate_essay", END)

app = workflow.compile()

###############################################################################
# 6) The main evaluate() function (with flatten fix)
###############################################################################
def evaluate(prompt, essay):
    """
    Evaluate a student's essay based on the given prompt using the StateGraph workflow.
    Returns the final LLM feedback text or an error if missing.
    """
    try:
        initial_state = {
            "prompt": prompt,
            "generation": None,
            "documents": [],
            "prompt_type": None,
            "student_essay": essay,
            "evaluation": None
        }

        evaluation_output = None
        # Run the workflow
        for output in app.stream(initial_state):
            evaluation_output = output  # final dictionary from the last node

        # 1) Some versions of langgraph store the final dict under the node name, e.g.:
        #    { "evaluate_essay": { "evaluation": "...text..." } }
        #    Flatten it if that's the case
        if evaluation_output and len(evaluation_output) == 1:
            # If there's exactly one key
            only_key = list(evaluation_output.keys())[0]
            # If that single key is "evaluate_essay" (the final node name)
            if only_key == "evaluate_essay":
                # Flatten: move that sub-dict up
                evaluation_output = evaluation_output[only_key]

        # 2) Now check if "evaluation" is present
        if evaluation_output and "evaluation" in evaluation_output and evaluation_output["evaluation"]:
            return evaluation_output["evaluation"]

        # If we still didn't find it
        return {
            "error": "No evaluation output generated",
            "details": "The workflow did not return a valid evaluation."
        }

    except Exception as e:
        raise RuntimeError(f"Error during evaluation: {e}")
