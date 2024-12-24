##############################
# evaluate_ai.py
##############################

import os
import openai
from dotenv import load_dotenv

from langchain.agents import Tool, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI as LegacyChatOpenAI
from langchain_openai import OpenAIEmbeddings as LegacyOpenAIEmbeddings

from PyPDF2 import PdfReader
from typing import List
from typing_extensions import TypedDict
from io import BytesIO

from langgraph.graph import END, StateGraph, START

# Pydantic for typed data models (if needed).
from pydantic import BaseModel

# Import your Pinecone setup function
from .pineconesetup import get_index


###############################################################################
# 1) Load environment variables and set up your API keys
###############################################################################
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set OpenAI API key for the openai library
openai.api_key = OPENAI_API_KEY

# Optionally, you could verify it’s loaded:
# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY not found. Please set it in your environment.")

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
        # CASE A: If query is None => skip embedding by using a dummy vector
        if query is None:
            # For text-embedding-ada-002, vector dimension is 1536
            dummy_vector = [0.0] * 1536
            results = index.query(
                vector=dummy_vector,
                top_k=100,
                include_metadata=True
            )
        else:
            # Normal embedding logic
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
                essay_metadata = metadata.get("essay_type_grad_receivede", "")

                # If prompt_type is None, skip the startswith filter
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
                        # If there's no (grade) pattern, store gracefully
                        filtered_results.append({
                            "text": metadata.get("text", ""),
                            "prompt_type": essay_metadata.strip(),
                            "grade": ""
                        })

        return filtered_results

    except Exception as e:
        raise RuntimeError(f"Error in embedding or querying Pinecone: {e}")


###############################################################################
# 3) Prompt Templates
###############################################################################
# Classification Prompt
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

# Main Evaluation Prompt
evaluation_prompt = PromptTemplate.from_template(
    """
You are an AP US History essay grader using the College Board's updated LEQ rubric from 2023. 
Your task is to evaluate a student's Long Essay Question (LEQ) strictly based on the rubric provided. 
All feedback, scores, and analysis must directly reference the rubric retrieved from the vector database. 
Do not introduce any criteria not explicitly included in the rubric.

The metadata structure for the essays you will use includes:
- essay_type_grad_receivede: A string in the format prompt_type(grade) (e.g., "causation(3)").
   - prompt_type: The type of the essay (e.g., "causation", "comparison", or "ccot").
   - grade: The grade received for the essay (e.g., "3", "6").
- text: The essay content.

You must take the entire structure into account:
1. Ensure you filter essays matching the correct prompt_type.
2. Include all grades of the identified type (e.g., both "causation(6)" and "causation(3)").
3. Use the text, prompt_type, and grade as context while evaluating.

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

formatting_prompt = PromptTemplate.from_template(
    """
Your sole task is to format the response to ensure it is well-structured and broken into clear sections. 
You must adhere to the following formatting rules when generating your response:

Use section headers to organize the response (e.g., "Contextualization", "Thesis / Claim", etc.).
Ensure each section is separated by a blank line.
Use bullet points or numbered lists where applicable for clarity.
Do not write long, dense paragraphs. Instead, break information into concise sentences or points.
If a score or feedback is involved, include it in the specified format, e.g., "[Score w/ feedback]".
Use consistent indentation for subsections.

For example, if evaluating an AP US History essay, your response should follow this exact structure:

---
Prompt type - Prompt Type

Contextualization (0-1 point):
[Score w/ feedback]

Thesis / Claim (0-1 point):
[Score w/ feedback]

Evidence (0-2 points):
Specific Evidence: [Score w/ feedback]
Evidence Supporting Argument: [Score w/ feedback]

Analysis and Reasoning (0-2 points):
Historical Reasoning: [Score w/ feedback]
Complex Understanding: [Score w/ feedback]

Total Score (out of 6):
[Score]

Feedback Summary:
[Concise summary of strengths, weaknesses, and areas for improvement.]

---

Ensure all sections are formatted cleanly with no extraneous text or deviations. Only provide the output in the required format without any additional explanations.

here is the essay and prompt type
{prompt}
{essay}
"""
)

# Extra classification prompt (not always used)
classify_prompt = PromptTemplate.from_template("""
You are a teaching assistant for an AP U.S. History class. Your task is to read the LEQ prompt provided by a student and classify it into one of the three main APUSH LEQ types:

Comparison: The prompt asks the student to compare and/or contrast historical developments, events, policies, or societies.
Causation: The prompt asks the student to explain causes and/or effects of historical events or developments.
Continuity and Change Over Time (CCOT): The prompt asks the student to analyze what changed and what remained the same over a specific time frame.

Instructions:
Carefully read the provided LEQ prompt.
Determine whether it is a "Comparison," "Causation," or "CCOT" prompt based on its wording and requirements.
Base your decision solely on the prompt text. Do not infer or consider anything outside the provided text.

Output:
Respond with only one word: "Comparison," "Causation," or "CCOT" depending on which category best matches the prompt.

Student’s Prompt: {prompt}
""")

# Smaller rubrics for partial grading
thesis_prompt = PromptTemplate.from_template(
    """Evaluate the thesis statement in the following essay based on the provided rubric and evaluation standards:

Rubric for Thesis / Claim:
Responds to the prompt with a historically defensible thesis or claim.
Establishes a line of reasoning.
Makes a claim that responds to the prompt (not merely restating or rephrasing it).
Consists of one or more sentences located in one place, either in the introduction or conclusion.
{rubric}

Evaluation Standards:
Ignore grammar and spelling errors as long as the meaning is clear.
Award 1 point only if the thesis meets all rubric criteria.
Do not award partial credit for incomplete or vague responses.
Be strict and apply no leniency.

Essay to Evaluate:
{essay}

Prompt Type: {prompt_type}

Output:
Score (0 or 1): Indicate whether the thesis earns the point.
Feedback: Provide a brief explanation justifying the score."""
)

contextualization_prompt = PromptTemplate.from_template(
    """
Evaluate the contextualization in the following essay based on the provided rubric and evaluation standards:

Rubric for Contextualization:
Describes a broader historical context relevant to the prompt. Relates the topic to broader historical events, developments, or processes before, during, or after the time frame of the question. Not awarded for merely a phrase or reference.
{rubric}

Evaluation Standards:
Ignore grammar and spelling errors as long as the meaning is clear.
Award 1 point only if the contextualization meets all rubric criteria.
Do not award partial credit for incomplete or vague contextualization.
Be strict and apply no leniency.
Contextualization must describe a broader historical event, development, or process relevant to the topic.
A single phrase or reference does not qualify as contextualization.

Essay to Evaluate:
{essay}

Prompt Type: 
{prompt_type}

Output:
Score (0 or 1): Indicate whether the contextualization earns the point.
Feedback: Provide a brief explanation justifying the score.
"""
)

evidence_prompt = PromptTemplate.from_template(
    """Evaluate the evidence and support for the argument in the following essay based on the rubric and evaluation standards:

**Rubric for Evidence & Support for Argument:**

1. **Evidence Usage (1 point):**
   - The response must provide at least two specific historical examples relevant to the topic of the prompt.
   - Examples must be concrete and historically accurate, such as specific events, dates, policies, or figures.
   - Vague references or generalizations do not satisfy this requirement.

2. **Evidence Supporting Argument (1 point):**
   - The response must use at least two specific and relevant pieces of historical evidence to support an argument in response to the prompt.
   - Evidence must be integrated into the argument, demonstrating how it supports the thesis.
   - Analysis and explanation are required to establish a clear connection between the evidence and the thesis.
   - Simply listing evidence without analysis does not meet this criterion.
{rubric}

**Evaluation Standards:**
- Ignore grammar and spelling errors as long as the meaning is clear.
- Award the first point (Evidence Usage) only if at least two specific historical examples are identified and relevant.
- Award the second point (Evidence Supporting Argument) only if:
  1. The first point (Evidence Usage) has been awarded.
  2. The response uses at least two specific and relevant pieces of historical evidence to support an argument, with clear and logical connections to the thesis.
- Be strict and apply no leniency.
- Do not award partial credit for incomplete, vague, or irrelevant responses.
- Ensure that all evidence is historically accurate and directly supports the argument.

**Essay to Evaluate:**
{essay}

**Prompt Type:** {prompt_type}

**Output:**
- Score (0, 1, or 2): Indicate the total points awarded for the Evidence & Support for Argument section.
- Feedback: Provide a brief explanation for the score, including:
  - Specific examples of evidence identified.
  - An assessment of how well the evidence supports the argument.
  - Reasons for any points not awarded.
"""
)

complexunderstanding_prompt = PromptTemplate.from_template(
    """Evaluate the analysis and reasoning in the following essay based on the rubric and evaluation standards. The evaluation must take into account the type of prompt (Comparison, Causation, or CCOT) when awarding points for historical reasoning.

**Rubric for Analysis and Reasoning:**

1. **Historical Reasoning (1 point):**
   - The response must use historical reasoning (e.g., comparison, causation, continuity and change over time) to frame or structure an argument that addresses the prompt.
   - To earn this point:
     - The reasoning should align with the specific type of prompt provided (e.g., comparison must compare and contrast, causation must address causes or effects, CCOT must highlight changes and continuities).
     - The reasoning may be uneven, imbalanced, or general, but it must clearly demonstrate an attempt to structure the argument around historical reasoning.

2. **Complex Understanding (1 point):**
   - The response must demonstrate a complex understanding of the historical development that is the focus of the prompt.
   - To earn this point, the essay must incorporate sophisticated argumentation and/or effective use of evidence. This can include:
     - **Exploring nuance** by analyzing multiple themes or perspectives.
     - **Explaining both sides of a concept**, such as similarity and difference, continuity and change, or multiple causes or effects.
     - **Drawing insightful connections** within and across historical periods or regions.
     - **Using evidence effectively** to highlight varying perspectives relevant to the prompt.
     - **Explaining at least four specific pieces of evidence** to support a nuanced or complex argument.
   - This understanding must be integrated into the argument and cannot be demonstrated through a single phrase or reference.
   - **This point can only be awarded if the Historical Reasoning point is awarded.**
{rubric}

**Evaluation Standards:**
- Ignore grammar and spelling errors as long as the meaning is clear.
- Award the Historical Reasoning point only if the response uses reasoning aligned with the specific type of prompt (Comparison, Causation, CCOT).
- Award the Complex Understanding point only if:
  1. The Historical Reasoning point has been awarded.
  2. The essay demonstrates sophisticated argumentation and/or nuanced use of evidence as outlined in the rubric.
- Be strict and apply no leniency. Do not award partial credit for incomplete, vague, or irrelevant responses.

**Essay to Evaluate:**
{essay}

**Prompt Type:** {prompt_type}

**Output:**
- Score (0, 1, or 2): Indicate the total points awarded for the Analysis and Reasoning section.
- Feedback: Provide a brief explanation for the score, including:
  - For Historical Reasoning: How the response used reasoning aligned with the specific type of prompt (e.g., comparison, causation, CCOT).
  - For Complex Understanding: How the response demonstrated a nuanced or sophisticated argument, referencing specific elements of the essay.
  - Reasons for any points not awarded.
"""
)

summation_prompt = PromptTemplate.from_template(
    """Based on the feedback and scores provided for each section of the essay, generate a comprehensive evaluation:

Section score and Feedback:
Thesis: {thesis_generation}

Contextualization: {contextualization_generation}

Evidence: {evidence_generation}

Analysis & Reasoning: {complexunderstanding_generation}

Total Score (0-6): Calculate the total score by summing up the scores included in the provided feedback.

Summary:
carefully sum up the scores from each section and sum up the feedback from each section.

Output Requirements:
Total Score (0-6): Include the calculated total score.
Final Feedback Summary: Provide a structured and detailed summary with:
Strengths
Weaknesses
dont do anything with just just mention this in the output. it is the prompt type. {prompt_type}
Suggestions for Improvement
"""
)

###############################################################################
# 4) LLM Setup
###############################################################################
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

tools = [
    Tool(
        name="get rubric and sample essays",
        func=lambda query: "\n\n".join(get_relevant_documents(query, None)),
        description="Retrieve relevant sections of the rubric and example essays for grading. Use the entire thing."
    )
]

from pydantic import BaseModel
from typing import List

###############################################################################
# 5) State Definition and Workflow
###############################################################################
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        prompt: The LEQ prompt from the student
        prompt_type: str for classification (Comparison, Causation, CCOT)
        student_essay: The student's actual essay text
        rubric: A list of docs from the rubric
        thesis_generation: Output from thesis evaluation
        contextualization_generation: Output from context evaluation
        evidence_generation: Output from evidence evaluation
        complexunderstanding_generation: Output from analysis reasoning
        summation: final combined output
    """
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


##############################
# Node Functions
##############################
def classify_prompt_node(state):
    """
    Node 1: Classify the student's LEQ prompt.
    """
    response = llm.invoke(classification_prompt.format(prompt=state["prompt"]))
    state["prompt_type"] = response.content.strip()
    return state


def fetch_rubric_node(state):
    query = "Leq Rubric"
    docs = get_relevant_documents(query, None)
    state["rubric"] = docs
    return state


def retrieve_essays_node(state):
    """
    Retrieve relevant docs from Pinecone based on 'prompt_type'.
    Optional if you want extra essay or rubric data for final LLM context.
    """
    prompt_type = state["prompt_type"]
    # If desired, fetch more docs with that prompt_type:
    docs = get_relevant_documents(None, prompt_type)
    # state["documents"] = docs  # If you want to store them
    return state


def thesis_grading_node(state):
    rubric = state["rubric"]
    essay = state["student_essay"]
    ptype = state["prompt_type"]
    response = llm.invoke(
        thesis_prompt.format(rubric=rubric, essay=essay, prompt_type=ptype)
    )
    state["thesis_generation"] = response.content
    return state


def contextualization_grading_node(state):
    rubric = state["rubric"]
    essay = state["student_essay"]
    ptype = state["prompt_type"]
    response = llm.invoke(
        contextualization_prompt.format(rubric=rubric, essay=essay, prompt_type=ptype)
    )
    state["contextualization_generation"] = response.content
    return state


def evidence_grading_node(state):
    rubric = state["rubric"]
    essay = state["student_essay"]
    ptype = state["prompt_type"]
    response = llm.invoke(
        evidence_prompt.format(rubric=rubric, essay=essay, prompt_type=ptype)
    )
    state["evidence_generation"] = response.content
    return state


def analysis_grading_node(state):
    rubric = state["rubric"]
    essay = state["student_essay"]
    ptype = state["prompt_type"]
    response = llm.invoke(
        complexunderstanding_prompt.format(
            rubric=rubric, essay=essay, prompt_type=ptype
        )
    )
    state["complexunderstanding_generation"] = response.content
    return state


def final_node(state):
    """
    The final node that composes a summation from all partial sections:
    thesis, contextualization, evidence, analysis.
    Stores the final text in state["summation"].
    """
    thesis = state["thesis_generation"] or ""
    cont = state["contextualization_generation"] or ""
    evidence = state["evidence_generation"] or ""
    complexu = state["complexunderstanding_generation"] or ""
    ptype = state["prompt_type"] or ""

    response = llm.invoke(
        summation_prompt.format(
            thesis_generation=thesis,
            contextualization_generation=cont,
            evidence_generation=evidence,
            complexunderstanding_generation=complexu,
            prompt_type=ptype
        )
    )
    state["summation"] = response.content
    return state


##############################
# Build the Workflow
##############################
workflow.add_node("classify_prompt", classify_prompt_node)
workflow.add_node("fetch_rubric", fetch_rubric_node)
workflow.add_node("retrieve_essays", retrieve_essays_node)
workflow.add_node("thesis_grading", thesis_grading_node)
workflow.add_node("contextualization_grading", contextualization_grading_node)
workflow.add_node("evidence_grading", evidence_grading_node)
workflow.add_node("analysis_grading", analysis_grading_node)
workflow.add_node("final_node", final_node)


workflow.add_edge(START, "classify_prompt")
workflow.add_edge("classify_prompt", "fetch_rubric")
workflow.add_edge("fetch_rubric", "retrieve_essays")
workflow.add_edge("retrieve_essays", "thesis_grading")
workflow.add_edge("thesis_grading", "contextualization_grading")
workflow.add_edge("contextualization_grading", "evidence_grading")
workflow.add_edge("evidence_grading", "analysis_grading")
workflow.add_edge("analysis_grading", "final_node")
workflow.add_edge("final_node", END)

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
            "prompt_type": None,
            "student_essay": essay,
            "rubric": [],
            "thesis_generation": None,
            "contextualization_generation": None,
            "evidence_generation": None,
            "complexunderstanding_generation": None,
            "summation": None
        }

        # Run the workflow
        final_output = None
        for output in app.stream(initial_state):
            final_output = output

        # Check if we got a final dictionary and "summation" is set
        if final_output and "summation" in final_output and final_output["summation"]:
            return final_output["summation"]

        # If we get here, we have no final text
        return {
            "error": "No evaluation output generated",
            "details": "The workflow did not return a valid final text."
        }

    except Exception as e:
        raise RuntimeError(f"Error during evaluation: {e}")
