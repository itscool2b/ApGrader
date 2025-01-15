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
client = OpenAI(api_key=OPENAI_API_KEY,max_retries=5)




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
    reflection: str
    isbsquestion: str

def essay_vision_node(state, max_retries=3):
    image_data = state.get('student_essay_image')
    if not image_data:
        state["student_essay_image"] = None
        return state

    if not image_data.startswith("data:"):
        image_data = f"data:image/jpeg;base64,{image_data}"

    for _ in range(max_retries):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract the text from the provided essay while ensuring it is grammatically correct, well-structured, and easy to read. Fix any spelling or typographical errors but do not alter the meaning, tone, or intent of the original content. If you can't recognize the image, respond exactly with: 'I can't recognize this image.' But obviously try your hardest. Your absolute hardest to do the task at hand.",
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
        if essay != "I can't recognize this image.":
            state["student_essay"] = essay
            return state

    state["student_essay"] = None
    return state

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

reflection = PromptTemplate.from_template("""
You are an AP Grader tasked with reflecting on your own grading outputs for an AP US History History SAQ (Short Answer Question). Your task is to extract the exact scores from the grading generations provided, ensure adherence to the appropriate rubric, and make changes only after thorough review. Your reflection must include:

Questions:
{questions}

Stimulus (if applicable):
{stimulus_description}

Student Essay:
{essay}

Generated Outputs:

Case 1 Evaluation: {case1_generation}

Case 2 Evaluation: {case2_generation}

Fact-Checking Feedback: {factchecking_generation} (if any)

Instructions:

Rubric Selection:

If case1_generation is provided, use the following rubric for grading:

Answer: Clearly and directly answers the question.

Cite: Provides at least one historically accurate piece of evidence relevant to the timeframe/context.

Explain: Explains how the evidence supports their answer.

Scoring: Assign 0 or 1 point for each subpart (A, B, C). Award 1 point only if all ACE criteria are met.

If case2_generation is provided, use the following rubric, integrating the stimulus:

Analyze the Stimulus: Identify and incorporate relevant details.

Answer: Directly answers the question.

Cite: Provides accurate evidence related to the timeframe/context.

Explain: Expands on how the evidence supports their answer.

Scoring: Assign 0 or 1 point for each subpart (A, B, C). Award 1 point only if all ACE criteria are met.

Stimulus Consideration:

If stimulus_description is None, disregard the stimulus in the evaluation.

Extract Scores:

Extract the explicit scores for each case evaluation directly from the generated outputs.

Do not interpolate or assume scores—use only the scores explicitly provided.

Ensure Rubric Adherence:

Carefully review the feedback and scores for each section to ensure alignment with every aspect of the rubric.

If changes to the scores are necessary, make them only after thorough review. Clearly explain the reason for any changes.

Feedback Verification and Enhancement:

Provide detailed, constructive, and actionable feedback explaining why the student earned or lost points and how they could improve.

Follow this format for each case where points were lost: "You put X and earned Y because Z. However, you could have said W to fully meet the rubric criteria."

Do not provide feedback for sections where full points were earned.

Changes and Final Summation:

Specify any changes made to scores, including the original score, the new score, and the reason for the change.

Output Format:

Section Scores:

Case 1 (0-3): Extracted score and explanation (if provided)

Feedback: Provide feedback only if less than full points were earned.

Case 2 (0-3): Extracted score and explanation (if provided)

Feedback: Provide feedback only if less than full points were earned.

Fact-Checking Feedback: Highlight any factual errors and their impact on scoring, if applicable. Provide constructive corrections in the format:

"You stated X, but the correct information is Y because Z."

Total Score (0-6):

Total Score: Sum the extracted scores explicitly provided in the generations. Reflect any changes here if scores were updated during review.

Changes Made:

Clearly specify any changes to scores, for example:

Case 1: "You put 2 but failed to fully answer the question, so the score was changed to 1."

Final Feedback Summary:

For each section where points were lost, provide feedback in the format: "You put X and earned Y because Z. However, you could have said W to earn the point."

Do not provide feedback for sections where full points were earned.

Be constructive and specific in guiding the student on how to improve.

Conclude with a summary of strengths and areas for improvement based on the rubric. Highlight exactly how the student can improve in future responses.

Do not include any extra commentary or user-friendly language. Output the results exactly as specified.



""")




def self_reflection(state):
    essay = state['student_essay']
    questions = state['questions']
    factchecking_generation = state['factchecking_generation']
    if state['case1_generation']:
        formatted_prompt = reflection.format(questions=questions,essay=essay, case1_generation=state['case1_generation'], case2_generation=None,factchecking_generation=factchecking_generation, stimulus_description=None)
        response = llm.invoke(formatted_prompt)
        state['reflection'] = response.content.strip()
        return state
    stim = state['stimulus_description']
    formatted_prompt = reflection.format(questions=questions,essay=essay, case1_generation=None, case2_generation=state['case2_generation'],factchecking_generation=factchecking_generation,stimulus_description=stim)
    response = llm.invoke(formatted_prompt)
    state['reflection'] = response.content.strip()
    return state


def summation_node(state):
    try:
        generation = state["case1_generation"] or state["case2_generation"]
        feedback = state.get("factchecking_generation", "")
        
        if not feedback:
            raise ValueError("Fact-checking feedback is missing.")
        s = state['student_essay']
        formatted_prompt = summation_prompt.format(generation=generation, factchecking=feedback,student_essay=s)
        response = llm.invoke(formatted_prompt) 
        concatenated = '\n\n This is more feedback (beta) - \n\n'
        state['summation'] = response.content.strip()
        final_output = f"{state['summation']}{concatenated}{state['reflection']}"
        return final_output
    except Exception as e:
        raise RuntimeError(f"Error in final_node: {e}")

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


def isbs(state):
    prompt = """
    Your task is to determine if the provided essay and prompt are valid AP European History (AP Euro) content.

    **Instructions:**
    - If the input is COMPLETELY random, nonsensical, or consists of unrelated words, symbols, or spam (e.g., keyboard mashing, random phrases with no structure), return ONLY the word **bs**.  
    - If the essay is poorly written, incomplete, or off-topic but still resembles an essay or has any relation to AP history, DO NOT flag it. These should simply receive a low score but are still valid submissions.  
    - If the content is an AP essay but for a different subject (e.g., APUSH, AP World), DO NOT flag it. Grade it as usual.  
    - If the essay is empty or has no meaningful content, return **bs**.  
    - For anything that is remotely related to AP European History or an essay format, even if it is bad, return ONLY the word **not**.

    **Examples:**  
    - Input: \"asldkjasd 1234 @!# random words\" → Output: **bs**  
    - Input: \"Discuss how the French Revolution changed Europe. It was bad and good.\" → Output: **not**  
    - Input: \"APUSH essay on the Civil War.\" → Output: **not**  

    **Prompt:** {prompt}  
    **Essay:** {essay}  

    Respond ONLY with **bs** or **not**. Do not provide explanations.
    """
    
    
    formatted_prompt = prompt.format(prompt=state['questions'], essay=state['student_essay'])
    
    
    response = llm.invoke(formatted_prompt)
    
    
    result = response.content.strip().lower()
    
    
    if "bs" in result:
        result = "bs"
    elif "not" in result:
        result = "not"
    else:
        result = "not"

    state["isbsquestion"] = result
    return state

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
        "student_essay_image": None,
        'isbsquestion': None,
        'reflection': None
    }
    state = isbs(state)
    if state['isbsquestion'] == 'bs':
        return 'submitt a valid response pls'
    state = vision_node(state)
    state = chapters(state)
    state = grading_node(state)
    state = factchecking_node(state)
    state = self_reflection(state)
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
        "student_essay_image": essay,
        'isbsquestion': None,
        'reflection': None
    }
    state = essay_vision_node(state)
    if state['student_essay'] == None:
            return 'we tried mutliple times to recognize your essay. You can try resubmitting but we have tried out best. Sorry'
    state = isbs(state)
    if state['isbsquestion'] == 'bs':
        return 'submitt a valid response pls'
    
    state = vision_node(state)
    state = chapters(state)
    state = grading_node(state)
    state = factchecking_node(state)
    state = self_reflection(state)
    state = summation_node(state)
    
    return state

