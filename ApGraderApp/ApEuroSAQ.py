import os
import json
import logging
import io
import base64
import re
from uuid import uuid4
from typing import List, Dict, Optional, Union, TypedDict, Any
from dotenv import load_dotenv
from PIL import Image
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import openai
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph, START
import ApGraderApp.p as p
from ApGraderApp.p import pc, setup_index, get_index

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your environment.")

openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)




index = get_index()

def retriever(query: str, top_k: int = 100) -> List[Dict]:
    """
    Retrieves relevant documents from Pinecone based on a query.

    Args:
        query (str): The search query.
        top_k (int): Number of top results to retrieve. Default is 100.

    Returns:
        List[Dict]: A list of retrieved documents with 'text' and 'metadata'.
    """
    try:
        
        if not query or not isinstance(query, str):
            raise ValueError(f"Query must be a non-empty string. Received: {query}")

        
        logging.debug(f"Query received: {query}")

        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )

        
        query_embedding = response.data[0].embedding

        
        logging.debug(f"Query embedding: {query_embedding}")

        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        
        logging.debug(f"Pinecone query results: {results}")

       
        return [
            {
                "text": match.get("metadata", {}).get("text", ""),
                "metadata": match.get("metadata", {})
            }
            for match in results.get("matches", [])
        ]
    except ValueError as ve:
        logging.error(f"Input Validation Error: {ve}")
        raise
    except Exception as e:
        logging.error(f"Error in retriever: {str(e)}")
        raise RuntimeError("Error in retriever function.") from e

from typing import Any, Dict
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")

case1 = PromptTemplate.from_template("""You are an AP European History SAQ grader. You will evaluate the provided essay response for a specific SAQ question. Each question may contain multiple subparts (e.g., A, B, C), and you must analyze and grade each subpart individually.


IMPORTANT NOTE - Never give a 0 unless the essay u recive is genuinley nothing and not related. If they have the absolute bare minimum just still give them a score.
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

case2 = PromptTemplate.from_template("""You are an AP European History SAQ grader. This question includes a stimulus (image, chart, or visual resource) that must be analyzed and integrated into the evaluation process. Your task is to grade each subpart of the SAQ (A, B, C, etc.) while taking the provided stimulus into account.


IMPORTANT NOTE - Never give a 0 unless the essay u receive is genuinley nothing and not related. If they have the absolute bare minimum just still give them a score.
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
Areas for Improvement: Suggest specific ways to improve their response.
How their essay relates to the stimulus be very specific                                   
                                     """)

ch_prompt = PromptTemplate.from_template("""
This is the student essay - {essay}

Write a query that I could put in a vector db to find relevant chapters to fact-check the content of the essay. I already have another prompt to fact-check, and I also pass in chapters.

So here should be your output:

**output**

A thorough query to find relevant chapters based on the student essay to fact-check. Your output should only consist of the query, that is it. that's it. it should be thorowough but nor paragraphgs.

""")

factchecking_prompt = PromptTemplate.from_template("""You are an expert AP European History essay fact-checker. Your task is to fact-check the content of a student's essay based on the chapters and topics retrieved from a vector database. Follow these instructions carefully:

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

summation_prompt = PromptTemplate.from_template("""You are tasked with summarizing the results of an SAQ grading session. Use the following guidelines to provide a final score, detailed feedback for each subpart (A, B, C, etc.), and a list of mistakes detected by the fact-checking node. Do not deduct points for misinformation, but include the mistakes in the feedback section for learning purposes.

### Steps for Summation:
1. **Input Data:**
   - {generation} - This includes all grading details for each subpart.
   - Fact-Checking Results: {factchecking}

2. **For Each Subpart:**
   - State the subpart (A, B, C, etc.).
   - Indicate whether the student earned a point.
   - Provide detailed feedback explaining why the student earned or did not earn the point.

3. **Total Score and General Feedback:**
   - Sum the scores for all subparts: **Total Score: total_score/3**.
   - Provide strengths: Highlight what the student did well (e.g., historical accuracy, strong reasoning, clear references to the stimulus).
   - Provide areas for improvement: Offer actionable suggestions for improving future responses (e.g., more detail, better integration, addressing all parts of the question).

4. **List Mistakes Detected by the Fact-Checking Node:**
   - Include factual inaccuracies identified by the fact-checking node.
   - Provide corrections for these inaccuracies.
   - Emphasize that these mistakes do not impact the total score.

### **Output Format**

#### **Feedback for Each Subpart:**
- **A:**
  - **Score:** X/1
  - **Feedback:** Explain why the student earned or did not earn this point.

- **B:**
  - **Score:** X/1
  - **Feedback:** Explain why the student earned or did not earn this point.

- **C:**
  - **Score:** X/1
  - **Feedback:** Explain why the student earned or did not earn this point.

#### **Total Score and General Feedback:**
- **Total Score:** X/3
- **Strengths:** Highlight key areas where the student performed well.
- **Areas for Improvement:** Provide actionable suggestions for improvement.

#### **Fact-Checking Node Feedback:**
- **Mistakes Identified:**
  - List all factual inaccuracies detected in the student’s response.
  - Include corrections for each mistake.
  - Emphasize that these mistakes do not impact the total score.
                                                
                                                """)

class Graphstate(TypedDict):
    questions: str
    case1_generation: str
    case2_generation: str
    student_essay: str
    factchecking_generation: str
    relevant_chapters: List[dict]
    summation: str
    image: Optional[Union[str, bytes]]
    stimulus_description: str

    student_essay_image: Optional[Union[str, bytes]]


def optional(state):

    try:
        image = state.get('student_essay_image')

def chapters(state):
    try:
        essay = state["student_essay"]
        formatted_prompt = ch_prompt.format(essay=essay)
        response = llm.invoke(formatted_prompt)
        query = response.content.strip()
        relevant_chapters = retriever(query)
        if not relevant_chapters:
            raise ValueError("No relevant chapters found.")
        state["relevant_chapters"] = relevant_chapters
        return state
    except Exception as e:
        raise ValueError(f"Error in chapters: {e}")




def vision_node(state):
    """
    Processes a Base64-encoded image using OpenAI's Vision API.
    Args:
        state (dict): The state containing Base64-encoded image data.
    Returns:
        dict: Updated state with stimulus_description.
    """
    try:
        
        image_data = state.get("image")
        if not image_data:
            state["stimulus_description"] = None
            return state

        
        if not image_data.startswith("data:"):
            image_data = f"data:image/jpeg;base64,{image_data}"  

        
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data},  
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        
        stimulus_description = response.choices[0].message.content
        state["stimulus_description"] = stimulus_description
        print(stimulus_description)
        return state

    except Exception as e:
        raise ValueError(f"Error in vision_node: {e}")
def grading_node(state):
    try:
        essay = state["student_essay"]  
        questions = state["questions"]
        stimulus_description = state.get("stimulus_description")
        if stimulus_description:
            formatted_prompt = case2.format(essay=essay, questions=questions, stimulus=stimulus_description)
            response = llm.invoke(formatted_prompt)
            state["case2_generation"] = response.content.strip()
        else:
            formatted_prompt = case1.format(essay=essay, questions=questions)
            response = llm.invoke(formatted_prompt)
            state["case1_generation"] = response.content.strip()
        return state
    except Exception as e:
        raise ValueError(f"Error in grading_node: {e}")

def factchecking_node(state):
    try:
        essay = state["student_essay"]
        chapters = state.get("relevant_chapters", [])
        if not chapters:
            raise ValueError("Fact-checking requires relevant chapters.")
        formatted_prompt = factchecking_prompt.format(essay=essay, chapters=chapters)
        response = llm.invoke(formatted_prompt)
        if not response or not response.content.strip():
            raise ValueError("Invalid response from fact-checking node.")
        state["factchecking_generation"] = response.content.strip()
        return state
    except Exception as e:
        raise ValueError(f"Error in factchecking_node: {e}")

def summation_node(state):
    try:
        generation = state["case1_generation"] or state["case2_generation"]
        feedback = state.get("factchecking_generation", "")
        
        if not feedback:
            raise ValueError("Fact-checking feedback is missing.")
        formatted_prompt = summation_prompt.format(generation=generation, factchecking=feedback)
        response = llm.invoke(formatted_prompt)
        state["summation"] = response.content.strip()
       
        return state
    except Exception as e:
        raise ValueError(f"Error in summation_node: {e}")


workflow = StateGraph(Graphstate)
workflow.add_node("vision_node", vision_node)
workflow.add_node("chapters", chapters)
workflow.add_node("grading_node", grading_node)
workflow.add_node("factchecking_node", factchecking_node)
workflow.add_node("summation_node", summation_node)
workflow.add_edge(START, "vision_node")
workflow.add_edge("vision_node", "chapters")
workflow.add_edge("chapters", "grading_node")
workflow.add_edge("grading_node", "factchecking_node")
workflow.add_edge("factchecking_node", "summation_node")
workflow.add_edge("summation_node", END)


app = workflow.compile()

def evaluateeurosaq(questions: str, essay: str, image: Optional[Union[str, bytes]]) -> str:
    state = {
        "questions": questions,
        "case1_generation": None,
        "case2_generation": None,
        "student_essay": essay,
        "student_essay_image": None,
        "factchecking_generation": None,
        "relevant_chapters": [],
        "summation": None,
        "image": image,
        "stimulus_description": None,
    }
    state = vision_node(state)
    state = chapters(state)
    state = grading_node(state)
    state = factchecking_node(state)
    state = summation_node(state)
    if "summation" in state and state["summation"]:
        return state["summation"]
    else:
        raise ValueError("Summation not found in the final state.")

def euro_saq_bulk_grading(questions: str, essay: str, image: Optional[Union[str, bytes]]) -> str:
    state = {
        "questions": questions,
        "case1_generation": None,
        "case2_generation": None,
        "student_essay": essay,
        "student_essay_image": None,
        "factchecking_generation": None,
        "relevant_chapters": [],
        "summation": None,
        "image": image,
        "stimulus_description": None,
    }
    state = vision_node(state)
    state = chapters(state)
    state = grading_node(state)
    state = factchecking_node(state)
    state = summation_node(state)
    if "summation" in state and state["summation"]:
        return state["summation"]
    else:
        raise ValueError("Summation not found in the final state.")


