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
# 3) Prompts
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
1. Ensure you filter essays matching the correct `prompt_type` (e.g., "causation").
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
- 1 point: Awarded only if the essay describes a broader historical context relevant to the prompt. The response must relate the topic of the prompt to broader historical events, developments, or processes that occur before, during, or continue after the time frame of the question.
- 0 points: Do not award points for vague, general, or unrelated contextual information. Mere passing references without clear connections to the argument are insufficient.

Thesis / Claim (0–1 point):
- 1 point: Awarded if the student responds to the prompt with a historically defensible thesis / claim that establishes a line of reasoning. The thesis must make a claim that responds to the prompt, rather than merely restating or rephrasing the prompt. The thesis must consist of one or more sentences located in one place, either in the introduction or the conclusion.
- 0 points: Do not award points for restatements of the prompt, overgeneralized statements, or claims lacking a clear line of reasoning.

Evidence (0–2 points):
- Specific Evidence (1 point): Award this point only if the essay clearly identifies at least two specific, relevant historical examples directly related to the topic of the prompt. Generalizations or broad statements without specific details do not merit this point.
- Evidence Supporting Argument (1 point): This second point can only be awarded if the essay has already earned the Specific Evidence point above. To earn this point, the essay must use at least two specific and relevant pieces of evidence to support a cohesive argument in response to the prompt. The connections between the evidence and the argument must be explicit and well-explained. If the essay fails to meet the requirements for the first evidence point, it cannot earn this second point.

Analysis and Reasoning (0–2 points):
- Historical Reasoning (1 point): Award this point only if the response demonstrates the use of at least one historical reasoning skill (e.g., comparison, causation, continuity and change) to frame or structure an argument that directly addresses the prompt. The reasoning may be uneven or imbalanced, and the evidence may be somewhat general, but the essay must clearly attempt to engage in a historical reasoning process.
- Complex Understanding (1 point): This second point can only be awarded if the essay has already earned the Historical Reasoning point above. To earn this point, the response must demonstrate a complex understanding of the historical development that is the focus of the prompt. This can be accomplished through sophisticated argumentation and/or effective use of evidence. Examples include:
  - Analyzing multiple variables or factors and explaining how they interact.
  - Considering diverse perspectives or interpretations.
  - Making connections across different historical periods, themes, or contexts.
  - Demonstrating insight that goes beyond a basic or superficial interpretation.
  - Makes connections from past time periods to the present day.

  If the response does not earn the Historical Reasoning point, it cannot receive the Complex Understanding point.

Output Format:
- Contextualization (0-1 point): [Score with feedback]
- Thesis / Claim (0-1 point): [Score with feedback]
- Evidence (0-2 points):
  - Specific Evidence: [Score with feedback]
  - Evidence Supporting Argument: [Score with feedback]
- Analysis and Reasoning (0-2 points):
  - Historical Reasoning: [Score with feedback]
  - Complex Understanding: [Score with feedback]
- Total Score (out of 6): [Score]

Feedback Summary:
Provide a realistic and strict summary of the essay’s strengths, weaknesses, and areas for improvement. Focus on alignment with the precise historical accuracy and analytical depth expected in AP US History essays.

Strict Grading Policy:
Always emphasize that the total score is out of 6 points, and apply no leniency. 
If the essay does not fulfill a point’s exact requirements, do not award partial credit.

Additional Instruction Considering Prompt Type:
- If "Comparison": The essay should meaningfully address similarities/differences and explain significance.
- If "Causation": The essay should explain causes/effects.
- If "CCOT": The essay should describe and analyze continuities/changes.

Final Note:
Ensure the total score is calculated as the sum of the points the student receives in the following categories: thesis, evidence, contextualization, and complex understanding and analysis.
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
        description="Retrieve relevant sections of the rubric and example essays for grading. Use the entire thing for grading."
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
        prompt_type: identified type (Comparison, Causation, CCOT)
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
    response = llm.invoke(classification_prompt.format(prompt=state["prompt"]))
    state["prompt_type"] = response.content.strip()
    return state  # Must return dict

def retrieve_documents(state):
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
    Final node: Evaluate the essay using the retrieved docs and store in state["evaluation"].
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
    state["evaluation"] = response.content
    return state  # Return the dict

workflow.add_node("classify_prompt", classify_prompt)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("evaluate_essay", evaluate_essay)

workflow.add_edge(START, "classify_prompt")
workflow.add_edge("classify_prompt", "retrieve_documents")
workflow.add_edge("retrieve_documents", "evaluate_essay")
workflow.add_edge("evaluate_essay", END)

app = workflow.compile()

###############################################################################
# 6) The main evaluate() function
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
        for output in app.stream(initial_state):
            evaluation_output = output  # final dictionary

        # 1) Check if the top-level dict has "evaluation"
        if evaluation_output and "evaluation" in evaluation_output and evaluation_output["evaluation"]:
            return evaluation_output["evaluation"]

        # 2) If the final node’s result is nested, e.g. "evaluate_essay": {"evaluation": "..."}
        if evaluation_output and "evaluate_essay" in evaluation_output:
            final_data = evaluation_output["evaluate_essay"]
            if "evaluation" in final_data and final_data["evaluation"]:
                return final_data["evaluation"]

        # If we still don't find it, fallback:
        return {
            "error": "No evaluation output generated",
            "details": "The workflow did not return a valid evaluation."
        }

    except Exception as e:
        raise RuntimeError(f"Error during evaluation: {e}")
