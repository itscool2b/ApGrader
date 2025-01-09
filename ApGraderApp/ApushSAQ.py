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
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0)

case1 = PromptTemplate.from_template(
"""You are an expert AP U.S History (APUSH) Short Answer Question (SAQ) grader. Each SAQ consists of multiple subparts (A, B, C), each requiring a concise response.

**SAQ Question:**
{questions}

**Student’s Response:**
{essay}

**Scoring Guidelines:**
Each subpart (A, B, C) is evaluated individually. Assign a score of 0 or 1 point per subpart:

- **1 Point:** awarded when the response meets the following criteria.
  - **Answer:** Clearly and directly answers the question.
  - **Cite:** Provides at least one historically accurate piece of evidence relevant to the timeframe/context of the question.
    - **Note:** A piece of evidence can include a specific event, policy, legal act, social movement, or accurate historical concept (e.g., "communism," "mercantilism") that is relevant to the timeframe and question.
    - **Benefit of the Doubt Clause:** If the student references a broader historical concept accurately tied to the question, you may count it as “evidence,” provided they connect it to their argument.
    - This still requires that the cited concept or event be **historically accurate** and contextually on-point (e.g., “communism after the Bolshevik Revolution” is valid; “communism in the American colonies” would not be).
  - **Explain:** Explains or expands how the evidence they cited supports their answer.
  
- **0 Points:** Does not meet one or more of the ACE criteria.

**Grading Process:**
1. **Evaluate Each Subpart (A, B, C):**
   - **Answer:** Does the response directly address the question?
   - **Cite:** Does the response cite at least one piece of evidence that is historically accurate and relevant to the timeframe/context of the question?
   - **Explain:** Does the student expand/elaborate on the evidence they cited to better support their argument?

2. **Assign Scores and Provide Feedback:**
   - If **1 Point Awarded:**
     - Clearly explain why the response meets all ACE criteria, referencing specific elements of the response and stimulus.
   - If **0 Points Awarded:**
     - Specify which ACE criteria were not met (Answer, Cite, Explain).
     - Provide constructive guidance on how to improve the response.

**Output Format:**
Feedback for Each Subpart:

- A (Score: X/1)
  - Feedback [Detailed feedback explaining why the response earned the point or why it did not.]

- B (Score: X/1)
  - Feedback [Detailed feedback explaining why the response earned the point or why it did not.]

- C (Score: X/1)
  - Feedback [Detailed feedback explaining why the response earned the point or why it did not.]

Total Score and Summary
- TOTAL SCORE: X/3
- Strengths: Highlight specific aspects of the responses that were well-executed.
- Areas for Improvement: Offer targeted suggestions for enhancing future responses.
- Originality Note: Comment on the appropriateness and originality of the evidence used, ensuring parts B and C reference events not included in the stimulus if required.

**Additional Grading Instructions:**
- Remember that SAQ's are meant to be very short, do not take off scores if the reponse is short. Responses do not need to elaborate and give examples of complex understanding and analysis.
- **Precision Over Presence:** Award points based on the quality and accuracy of the response, not merely the presence of keywords or events.
- **Alignment with Historians' Arguments:** Ensure explanations reflect the specific arguments and interpretations relevant to the question.
- **Constructive Feedback:** Provide feedback that guides the student on improving accuracy, relevance, and depth of analysis.
- **Strict Binary Scoring:** Only assign 1 point if all ACE criteria are fully met; otherwise, assign 0 points.
""")

case2 = PromptTemplate.from_template(
"""You are an expert AP U.S History (APUSH) Short Answer Question (SAQ) grader. Each SAQ consists of multiple subparts (A, B, C), each requiring a concise response. SAQs include a stimulus (image, chart, or visual resource) that must be analyzed and integrated into the grading process.

**Stimulus Description:**
{stimulus}

**SAQ Question:**
{questions}

**Student’s Response:**
{essay}

**Scoring Guidelines:**
Each subpart (A, B, C) is evaluated individually. Assign a score of 0 or 1 point per subpart:

- **1 Point:** awarded when the response meets the following criteria.
  - **Answer:** Clearly and directly answers the question.
  - **Cite:** Provides at least one historically accurate piece of evidence relevant to the timeframe/context of the question.
    - **Note:** A piece of evidence can include a specific event, policy, legal act, social movement, or accurate historical concept (e.g., "communism," "mercantilism") that is relevant to the timeframe and question.
    - **Benefit of the Doubt Clause:** If the student references a broader historical concept accurately tied to the question, you may count it as “evidence,” provided they connect it to their argument.
    - This still requires that the cited concept or event be historically accurate and contextually on-point (e.g., “communism after the Bolshevik Revolution” is valid; “communism in the American colonies” would not be).
  - **Explain:** Explains or expands how the evidence they cited supports their answer.
  
- **0 Points:** Does not meet one or more of the ACE criteria.

**Grading Process:**
1. **Analyze the Stimulus:**
   - Identify key details and determine its relevance to the question.
   
2. **Evaluate Each Subpart (A, B, C):**
   - **Answer:** Does the response directly address the question?
   - **Cite:** Does the response cite at least one piece of evidence that is historically accurate and relevant to the timeframe/context of the question?
   - **Explain:** Does the student expand/elaborate on the evidence they cited to better support their argument?

3. **Assign Scores and Provide Feedback:**
   - If **1 Point Awarded:**
     - Clearly explain why the response meets all ACE criteria, referencing specific elements of the response and stimulus.
   - If **0 Points Awarded:**
     - Specify which ACE criteria were not met (Answer, Cite, Explain).
     - Provide constructive guidance on how to improve the response.

**Output Format:**
Feedback for Each Subpart:

- A (Score: X/1)
  - Feedback [Detailed feedback explaining why the response earned the point or why it did not.]

- B (Score: X/1)
  - Feedback [Detailed feedback explaining why the response earned the point or why it did not.]

- C (Score: X/1)
  - Feedback [Detailed feedback explaining why the response earned the point or why it did not.]

Total Score and Summary
- TOTAL SCORE: X/3
- Strengths: Highlight specific aspects of the responses that were well-executed.
- Areas for Improvement: Offer targeted suggestions for enhancing future responses.
- Originality Note: Comment on the appropriateness and originality of the evidence used, ensuring parts B and C reference events not included in the stimulus if required.

**Additional Grading Instructions:**
- Remember that SAQ's are meant to be very short, do not take off scores if the reponse is short. Responses do not need to elaborate and give examples of complex understanding and analysis.
- **Precision Over Presence:** Award points based on the quality and accuracy of the response, not merely the presence of keywords or events.
- **Alignment with Historians' Arguments:** Ensure explanations reflect the specific arguments and interpretations relevant to the question.
- **Constructive Feedback:** Provide feedback that guides the student on improving accuracy, relevance, and depth of analysis.
- **Strict Binary Scoring:** Only assign 1 point if all ACE criteria are fully met; otherwise, assign 0 points.
""")
ch_prompt = PromptTemplate.from_template("""
This is the student essay - {essay}

Write a query that I could put in a vector db to find relevant chapters to fact-check the content of the essay. I already have another prompt to fact-check, and I also pass in chapters.

So here should be your output:

**output**

A thorough query to find relevant chapters based on the student essay to fact-check. Your output should only consist of the query, that is it. that's it. it should be thorowough but nor paragraphgs.

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

### **Output Format**

Feedback for Each Subpart:

- A (Score: X/1)
  - Feedback [Detailed feedback explaining why the response earned the point or why it did not.]

- B (Score: X/1)
  - Feedback [Detailed feedback explaining why the response earned the point or why it did not.]

- C (Score: X/1)
  - Feedback: [Detailed feedback explaining why the response earned the point or why it did not.]

Total Score and Summary
- TOTAL SCORE: X/3
- Strengths: Highlight specific aspects of the responses that were well-executed.
- Areas for Improvement: Offer targeted suggestions for enhancing future responses.
- Originality Note: Comment on the appropriateness and originality of the evidence used, ensuring parts B and C reference events not included in the stimulus if required.

**Additional Grading Instructions:**
- **Precision Over Presence:** Award points based on the quality and accuracy of the response, not merely the presence of keywords or events.
- **Alignment with Historians' Arguments:** Ensure explanations reflect the specific arguments and interpretations relevant to the question.
- **Constructive Feedback:** Provide feedback that guides the student on improving accuracy, relevance, and depth of analysis.
- **Strict Binary Scoring:** Only assign 1 point if all ACE criteria are fully met; otherwise, assign 0 points.
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
    student_essay_image: Optional[Union[str,bytes]]

def essay_vision_node(state):

    try:
        image_data = state.get('student_essay_image')
        if not image_data:
            state["student_essay_image"] = None
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
                            "text": "Extract the text from the provided essay while ensuring it is grammatically correct, well-structured, and easy to read. Fix any spelling or typographical errors but do not alter the meaning, tone, or intent of the original content. Group ideas logically to improve flow, but preserve the original structure as much as possible. Avoid including irrelevant page numbers, headings, or formatting instructions unless they are part of the essay's content. Pay special attention to preserving references to documents (e.g., '(Doc 7)', '(Doc 5)'), ensuring they are included exactly as written and placed correctly in the text.  Group ideas logically to improve flow, but do not omit or rephrase any document referencesOutput the text only with nothing else.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data},  
                        },
                    ],
                }
            ],
            max_tokens=16384,
            temperature=0,
        )

        
        essay = response.choices[0].message.content
        state["student_essay"] = essay
        print(essay)
        return state

    except Exception as e:
        raise ValueError(f"Error in vision_node: {e}")

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
                            "text": "Give a full complete and detailed description and analysis of the stimulus or image. Be very thorough.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data},  
                        },
                    ],
                }
            ],
            max_tokens=3000,
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
        s = state['student_essay']
        formatted_prompt = summation_prompt.format(generation=generation, factchecking=feedback,student_essay=s)
        response = llm.invoke(formatted_prompt)
        sum = response.content.strip()
        t = ' \n \nThis is the text that our Ai was able to extract from the image of your essay. If you feel the score is innacurate, please make sure that the Ai has accurately analyzed and extracted the text from the essay. If not, please make the needed edits to the extracted text and paste it into our text submission for accurate grading: \n \n '
        full = sum + t + s
       
        return full
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

def evaluate1(questions: str, essay: str, image: Optional[Union[str, bytes]]) -> str:
    state = {
        "questions": questions,
        "case1_generation": None,
        "case2_generation": None,
        "student_essay": essay,
        "factchecking_generation": None,
        "relevant_chapters": [],
        "summation": None,
        "image": image,
        "stimulus_description": None,
        "student_essay_image": None
    }
    state = vision_node(state)
    state = chapters(state)
    state = grading_node(state)
    state = factchecking_node(state)
    state = summation_node(state)
    
    return state

def evaluate11(questions: str, essay, image: Optional[Union[str, bytes]]) -> str:
    state = {
        "questions": questions,
        "case1_generation": None,
        "case2_generation": None,
        "student_essay": None,
        "factchecking_generation": None,
        "relevant_chapters": [],
        "summation": None,
        "image": image,
        "stimulus_description": None,
        "student_essay_image": essay
    }
    state = essay_vision_node(state)
    state = vision_node(state)
    state = chapters(state)
    state = grading_node(state)
    state = factchecking_node(state)
    state = summation_node(state)
    
    return state

