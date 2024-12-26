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

from typing import List, Dict, Optional, Union, TypedDict

from langgraph.graph import END, StateGraph, START

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your environment.")

openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

index = get_index()

def retriever(query: str, top_k: int = 100) -> List[Dict]:
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

case1 = PromptTemplate.from_template("""You are an AP U.S. History SAQ grader. You will evaluate the provided essay response for a specific SAQ question. Each question may contain multiple subparts (e.g., A, B, C), and you must analyze and grade each subpart individually.

Steps for Grading:
Read the Provided SAQ Question and Essay Response:

Question:
{questions}

Student’s Response:
{essay}

Analyze the Structure:

Identify how many subparts (A, B, C, etc.) the question contains.
Evaluate each subpart based on the task verb and requirements of the question.
Grade Each Subpart (0–1 Point Per Subpart):

Assign 1 point if the subpart is fully accurate, relevant, and sufficiently detailed.
Assign 0 points if the subpart is vague, irrelevant, incomplete, or historically inaccurate.
Provide Feedback for Each Subpart:

If the response earns the point: Explain why it is correct.
If the response does not earn the point: Explain what is missing, incorrect, or vague.
Summarize the Total Score:

Add up the points earned across the subparts (e.g., 0–3 points for a 3-part question).
Provide a summary at the end, highlighting strengths and areas for improvement.
Criteria for Common Task Verbs
Identify: Name or state a historically accurate fact relevant to the question.
Describe: Provide specific details about a historical fact, event, or concept.
Explain: Analyze cause/effect, context, or significance of the topic in question.
Compare: Discuss both similarities and/or differences with clear connections to the question.
Analyze: Break down the significance, impact, or meaning of a historical development.
Output Format for Grading
Feedback for Each Subpart:

A: (Score: X/1) Feedback...
B: (Score: X/1) Feedback...
C: (Score: X/1) Feedback...
Total Score and Summary:

Total Score: X/Y (e.g., 2/3 for a 3-part question).
Strengths: Highlight what the student did well.
Areas for Improvement: Suggest specific ways to improve their response.

""")

case2 = PromptTemplate.from_template("""You are an AP U.S. History SAQ grader. This question includes a stimulus (image, chart, or visual resource) that must be analyzed and integrated into the evaluation process. Your task is to grade each subpart of the SAQ (A, B, C, etc.) while taking the provided stimulus into account.

Steps for Grading:
Analyze the Provided Stimulus:

Stimulus Description:
{stimulus}

Carefully review the stimulus and identify its key details (e.g., captions, labels, or visual elements).
Understand how the stimulus relates to the historical context and themes referenced in the question.
Read the SAQ Question and Student Response:

Question:
{questions}

Student’s Response:
{essay}

Grade Each Subpart (A, B, C, etc.):

For Subparts Requiring Stimulus Analysis:
Evaluate if the student accurately references and interprets the details or themes from the stimulus.
Award points for correct and relevant integration of the stimulus with historical context or reasoning.
For Other Subparts:
Grade based on the general rubric for historical evidence (accuracy, relevance, depth).
Provide Feedback for Each Subpart:

If the response earns the point: Explain why it is correct, including references to the stimulus if required.
If the response does not earn the point: Explain what is missing, incorrect, or vague, including any missed connections to the stimulus.
Summarize the Total Score:

Add up the points earned across all subparts (e.g., 0–3 for a 3-part SAQ).
Provide a summary of strengths and areas for improvement.
Grading Criteria
For Subparts Requiring the Stimulus:
Accuracy:

Does the response correctly reference details or themes from the stimulus?
Relevance:

Does the response use the stimulus to address the question directly?
Integration with Broader Knowledge:

Does the response connect observations from the stimulus to broader historical themes or developments?
General Criteria for All Subparts:
Identify: Name or state a correct, relevant historical fact.
Describe: Provide specific details about a historical event or concept.
Explain: Analyze cause/effect, context, or significance.
Compare: Discuss similarities and/or differences explicitly.
Analyze: Break down the meaning, impact, or significance of the stimulus or broader topic.
Output Format for Grading
Stimulus Reference:

Explanation of how the stimulus relates to the question.
Feedback for Each Subpart:

A: (Score: X/1) Feedback...
B: (Score: X/1) Feedback...
C: (Score: X/1) Feedback...
Total Score and Summary:

Total Score: X/Y (e.g., 2/3 for a 3-part question).
Strengths: Highlight what the student did well.
Areas for Improvement: Suggest specific ways to improve their response.""")

ch_prompt = PromptTemplate.from_template("""
This is the student essay - {essay}

Write a query that I could put in a vector db to find relevant chapters to fact-check the content of the essay. I already have another prompt to fact-check, and I also pass in chapters.

So here should be your output:

**output**

A thorough query to find relevant chapters based on the student essay to fact-check. Your output should only consist of the query, that is it. that's it

""")

factchecking_prompt = PromptTemplate.from_template("""You are an expert AP US History essay fact-checker. Your task is to fact-check the content of a student's essay based on the chapters and topics retrieved from a vector database. Follow these instructions carefully:

Fact-Check the Essay: Review the essay for historical accuracy. Cross-reference claims and information in the essay with the content provided in the {chapters} from the vector database. Focus on ensuring the essay aligns with the correct historical events, dates, figures, and interpretations.

Here is the essay - {essay}

Feedback on Mistakes:

If the student makes a factual error, do not deduct points. Instead, provide constructive feedback.
Highlight the specific part of the essay where the mistake occurs.
Explain the correct historical information in a clear and concise manner.
Use positive and educational language to encourage learning.
General Comments:

Summarize the overall accuracy of the essay.
Mention areas where the student demonstrated accurate historical understanding and areas for improvement.
Example Structure for Your Feedback:
Identified Mistake: "In your essay, you stated that [incorrect information]. However, according to [chapter/topic], the correct information is [correct information]."
General Accuracy: "Overall, your essay is accurate in its portrayal of [topic], but keep an eye on [specific areas]."
Focus on being supportive and informative. Your goal is to help the student learn and improve their historical understanding without penalizing them for mistakes.""")

summation_prompt = PromptTemplate.from_template("""You are tasked with summarizing the results of an SAQ grading session. Use the following guidelines to provide a final score, general feedback, and a list of mistakes detected by the fact-checking node. Do not deduct points for misinformation, but include the mistakes in the feedback section.

Steps for Summation:
Input Data:
{generation} - this includes all grading
Sum the scores for all subparts:
Total Score: total_score/3
General Feedback on Student Performance:

Highlight what the student did well (e.g., historical accuracy, strong reasoning, clear references to the stimulus).
Provide suggestions for improvement (e.g., more detail, better integration of the stimulus, addressing all parts of the question).
List Mistakes Detected by the Fact-Checking Node:

Include any factual inaccuracies from the student response, as identified by the fact-checking node.
Emphasize that these mistakes do not impact the score but should be reviewed for learning purposes.
Output Format
Total Score and General Feedback:
{factchecking} - this is factchecking. Put it in the feedback
Total Score: X/Y
Strengths: Summarize key areas where the student performed well.
Areas for Improvement: Provide actionable suggestions for improvement.
Fact-Checking Node Feedback:

Mistakes Identified:
List all factual inaccuracies detected in the student’s response, with corrections.""")

class Graphstate(TypedDict):
    questions: str
    case1_generation: str
    case2_generation: str
    student_essay: str
    factchecking_generation: str
    relevant_chapters: List[dict]
    summation: str
    image: Optional[Union[str, bytes]]

def chapters(state):
    essay = state["student_essay"]
    formatted_prompt = ch_prompt.format(essay=essay)
    query = llm.invoke(formatted_prompt)
    state["relevant_chapters"] = retriever(query)
    return state

def grading_node(state):
    essay = state["student_essay"]
    questions = state["questions"]
    
    if state["image"] is None:
        formatted_prompt = case1.format(essay=essay, questions=questions)
        response = llm.invoke(formatted_prompt)
        state["case1_generation"] = response.content.strip()
    else:
        stimulus = state["image"]
        formatted_prompt = case2.format(essay=essay, questions=questions, stimulus=stimulus)
        response = llm.invoke(formatted_prompt)
        state["case2_generation"] = response.content.strip()

    return state

def factchecking_node(state):
    essay = state["student_essay"]
    chapters = state["relevant_chapters"]
    formatted_prompt = factchecking_prompt.format(essay=essay, chapters=chapters)
    response = llm.invoke(formatted_prompt)
    state["factchecking_generation"] = response.content.strip()

    return state

def summation_node(state):
    if state["case1_generation"]:
        generation = state["case1_generation"]
    else:
        generation = state["case2_generation"]

    feedback = state["factchecking_generation"]
    formatted_prompt = summation_prompt.format(generation=generation, feedback=feedback)
    response = llm.invoke(formatted_prompt)
    state["summation"] = response.content.strip()

    return state

workflow = StateGraph(Graphstate)

workflow.add_node("chapters", chapters)
workflow.add_node("grading_node", grading_node)
workflow.add_node("factchecking_node", factchecking_node)
workflow.add_node("summation_node", summation_node)

workflow.add_edge(START, "chapters")
workflow.add_edge("chapters", "grading_node")
workflow.add_edge("grading_node", "factchecking_node")
workflow.add_edge("factchecking_node", "summation_node")
workflow.add_edge("summation_node", END)

app = workflow.compile()

def evaluate1(questions, essay, image):
    state = {
        "questions": questions,
        "case1_generation": None,
        "case2_generation": None,
        "student_essay": essay,
        "factchecking_generation": None,
        "relevant_chapters": [],
        "summation": None,
        "image": image
    }

    app.run(state)

    if "summation" in state and state["summation"]:
        return state["summation"]
    else:
        raise ValueError("Summation not found in the final state.")
