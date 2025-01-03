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

OPENAI_API_KEY = os.getenv('OPENAI_APi_KEY')

openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

index = get_index()

def retriever(query: str, top_k: int = 100) -> List[Dict]:
    """
    Generalized function to retrieve relevant documents from Pinecone based on a query.

    Args:
        query (str): The search query.
        top_k (int): Number of top results to retrieve. Default is 100.

    Returns:
        List[Dict]: A list of retrieved documents with 'text' and 'metadata'.
    """
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



classification_prompt = PromptTemplate.from_template(
    """
        You are a highly accurate and strict teaching assistant for an AP U.S. History class. Your task is to read the LEQ prompt provided by a student and determine which of the three main APUSH LEQ types it falls under:
        
Comparison: The prompt asks the student to compare and/or contrast historical developments, events, policies, or societies.
Causation: The prompt asks the student to explain causes and/or effects of historical events or developments.
Continuity and Change Over Time (CCOT): The prompt asks the student to analyze what changed and what remained the same over a particular time frame.

        Instructions:
        
Read the provided LEQ prompt carefully.
Identify whether the prompt is a Comparison, Causation, or CCOT prompt.
Respond with only one of the three exact words: "Comparison", "Causation", or "CCOT". Do not include any additional text, explanations, or characters. Should be one word

        Student’s Prompt to Classify: {prompt}

        Your Response:
"""
)

thesis_prompt = PromptTemplate.from_template(
    """Evaluate the thesis statement in the following essay based on the provided rubric and evaluation standards:

Rubric for Thesis / Claim:
Responds to the prompt with a historically defensible thesis or claim that establishes a line of reasoning.
Makes a claim that responds directly to the prompt, rather than merely restating or rephrasing it.
Consists of one or more sentences located in one place, either in the introduction or conclusion of the essay.

Evaluation Standards:
Ignore grammar and spelling errors as long as the meaning is clear.
Award 1 point only if the thesis fully meets all rubric criteria.
Do not award partial credit for incomplete, vague, or partially correct theses.
Be strict and apply no leniency; adhere strictly to the rubric.

Essay to Evaluate:
{essay}

Prompt Type: {prompt_type}

Output:
Score (0 or 1): Indicate whether the thesis earns the point.
Feedback: Provide a brief explanation justifying the score. Highlight which criteria were met or not met.
"""
)
contextualization_prompt = PromptTemplate.from_template(
    """Evaluate the contextualization in the following essay based on the provided rubric and evaluation standards:

Rubric for Contextualization:
Describes a broader historical context relevant to the prompt.
Relates the topic to broader historical events, developments, or processes that occurred before, during, or after the time frame of the question.
Not awarded for merely a phrase or reference.

Evaluation Standards:
Ignore grammar and spelling errors as long as the meaning is clear.
Award 1 point only if the contextualization fully meets all rubric criteria.
Do not award partial credit for incomplete, vague, or partially correct contextualization.
Be strict and apply no leniency; adhere strictly to the rubric.
Contextualization must describe a broader historical event, development, or process relevant to the topic.
A single phrase or reference does not qualify as contextualization.

Essay to Evaluate:
{essay}

Prompt Type: {prompt_type}

Output:
Score (0 or 1): Indicate whether the contextualization earns the point.
Feedback: Provide a brief explanation justifying the score. Highlight which criteria were met or not met.
""")

evidence_prompt = PromptTemplate.from_template(
    """You are an AP U.S. History (APUSH) DBQ grader. Your task is to evaluate the Evidence section of a student's essay based on the provided rubric. Follow the instructions carefully to ensure accurate and consistent grading.

**DBQ Evidence Rubric**:

1. **First Evidence Point (1 Point):**
   - **Requirement:** Accurately describe the content of at least three documents and address the topic of the prompt.
   - **Key Detail:** Simply quoting the documents doesn’t count; you need to explain what the document is saying in your own words and relate it to the prompt.
   - **Important Note:** If this point is not earned, the second evidence point cannot be earned, as it builds on this one.

2. **Second Evidence Point (1 Point):**
   - **Requirement:** Use at least four documents to support an argument in response to the prompt.
   - **Key Detail:** This means you’re not just describing the documents, but actively using them as evidence to back up your thesis or main argument.
   - **Dependency:** Requires the first evidence point to be earned.

3. **HIPP (Contextual Analysis) Point (1 Point):**
   - **Requirement:** For at least two documents, explain how or why the document’s historical situation, intended audience, point of view (POV), and/or purpose (HIPP) is relevant to the argument.
   - **Independence:** This point is separate from the first two. It can be earned even if the first two points are not.

**Evaluation Standards**:
- **Ignore grammar and spelling errors** as long as the meaning is clear.
- **Award points only if the criteria are fully met**; do not award partial credit.
- **Be strict and apply no leniency**; adhere strictly to the rubric.
- **Contextual Analysis (HIPP)** must clearly connect the document features to the argument.
- **Descriptions and usage** of documents must be in the student's own words and relevant to the prompt.

**Document Descriptions**:
- **Document 1 Description:** {doc1}
- **Document 2 Description:** {doc2}
- **Document 3 Description:** {doc3}
- **Document 4 Description:** {doc4}
- **Document 5 Description:** {doc5}
- **Document 6 Description:** {doc6}
- **Document 7 Description:** {doc7}

**Essay to Evaluate**:
{essay}

**Prompt Type**: {prompt_type}

**Instructions**:
1. **Analyze the Use of Documents**:
   - Review each document description (Doc 1 to Doc 7) to determine how accurately and effectively they are described in the essay.
   - Assess whether the student has explained the content of each document in their own words and related it to the prompt.
   - Identify which documents are used to support the argument and how they contribute to the thesis.

2. **Evaluate Each Evidence Point**:
   - Evaluate and grade the essay based on the DBQ Evidence Rubric given to you above.

3. **Assign Scores and Provide Feedback**:
   - **Total Score (0–3)**: Sum the points earned across the three Evidence points.
   - **Breakdown**:
     - **First Evidence Point (1 Point)**: (Explain if criteria are met)
     - **Second Evidence Point (1 Point)**: (Explain if criteria are met)
     - **HIPP Point (1 Point)**: (Explain if criteria are met)
   - **Overall Feedback**: Summarize the strengths and areas for improvement regarding the use of evidence in the essay. Include specific references to how documents were used or could be better utilized.

**Output**:
- **Total Score (0–3)**: Provide a total score.
- **Breakdown**:
  - **First Evidence Point (1 Point)**: Yes/No
    - *Explanation:* Brief explanation for why the point was earned or not.
  - **Second Evidence Point (1 Point)**: Yes/No
    - *Explanation:* Brief explanation for why the point was earned or not.
  - **HIPP Point (1 Point)**: Yes/No
    - *Explanation:* Brief explanation for why the point was earned or not.
- **Overall Feedback**: Provide a summary of the strengths and areas for improvement."""
)

evidence_beyond_prompt = PromptTemplate.from_template(
    """You are an AP U.S. History (APUSH) DBQ grader. Your task is to evaluate the "Evidence Beyond the Documents" point of a student's essay based on the provided rubric. Follow the instructions carefully to ensure accurate and consistent grading.

**DBQ Evidence Beyond the Documents Rubric**:

1. **Requirement:**
   - The student must provide specific evidence that is not in the provided documents.
   - This evidence must:
     - Be relevant to the argument.
     - Be fully described, not just a simple phrase or reference.
     - Be different from the evidence used to earn the contextualization point.

**Evaluation Standards**:
- **Ignore grammar and spelling errors** as long as the meaning is clear.
- **Award 1 point only** if the "Evidence Beyond the Documents" meets all rubric criteria.
- **Do not award partial credit** for incomplete, vague, or partially correct responses.
- **Be strict and apply no leniency**; adhere strictly to the rubric.
- **Evidence must be specific**, fully described, **relevant**, and **distinct** from contextualization evidence.

**Document Descriptions**:
- **Document 1 Description:** {doc1}
- **Document 2 Description:** {doc2}
- **Document 3 Description:** {doc3}
- **Document 4 Description:** {doc4}
- **Document 5 Description:** {doc5}
- **Document 6 Description:** {doc6}
- **Document 7 Description:** {doc7}


**Essay to Evaluate**:
{essay}

**Prompt Type**: {prompt_type}

**Instructions**:
1. **Identify Evidence Beyond the Documents**:
   - Look for specific evidence in the essay that is **not included** in the provided documents.
   - Determine if the evidence is **fully described** and **explained**, not just mentioned.
   - Check if the evidence is **relevant** to the argument and **supports** the thesis.
   - Ensure the evidence is **different** from the evidence used for contextualization.

2. **Evaluate the Evidence Beyond the Documents Point**:
   - **Award 1 point** if all the following criteria are met:
     - Specific evidence not in the documents is provided.
     - The evidence is fully described and explained.
     - The evidence is relevant to the argument.
     - The evidence is different from contextualization evidence.
   - **Do not award the point** if any of the above criteria are not met.

3. **Provide Feedback**:
   - **If the point is earned**:
     - Specify which outside evidence was used by quoting the relevant part of the student's essay.
     - Explain how this evidence meets the rubric criteria.
   - **If the point is not earned**:
     - Briefly explain which criteria were not met without mentioning specific documents.

**Output**:
- **Score (0 or 1)**: Indicate whether the "Evidence Beyond the Documents" earns the point.
- **Feedback**: Provide a brief explanation justifying the score.
  - **If earned**: Quote the specific outside evidence used and explain how it meets the criteria.
  - **If not earned**: Explain why the criteria were not met without mentioning specific documents."""
)

complex_understanding_prompt = PromptTemplate.from_template(
    """You are an AP U.S. History (APUSH) DBQ grader. Your task is to evaluate the "Complex Understanding" point of a student's essay based on the provided rubric. Follow the instructions carefully to ensure accurate and consistent grading.

**DBQ Complex Understanding Rubric**:

1. **Sophisticated Argumentation:**
   - **Nuanced Argument:** Explores multiple themes or perspectives.
   - **Balanced Comparison:** Explains both sides of a comparison (e.g., similarity and difference, continuity and change).
   - **Broader Connections:** Makes connections across time periods, themes, or regions.
   
2. **Effective Use of Evidence:**
   - **Comprehensive Document Usage:** Uses all seven documents effectively to support the argument.
   - **HIPP Analysis:** Includes effective HIPP (Historical Situation, Intended Audience, Point of View, Purpose) analysis for at least four documents.
   - **Incorporates Outside Evidence:** Uses both documents and outside evidence to add depth to the argument.

**Evaluation Standards**:
- **Ignore grammar and spelling errors** as long as the meaning is clear.
- **Award 1 point only** if the "Complex Understanding" meets all rubric criteria.
- **Do not award partial credit** for incomplete, vague, or partially correct responses.
- **Be strict and apply no leniency**; adhere strictly to the rubric.
- **Demonstrate sophistication** through nuanced argumentation and/or effective use of evidence.
- **Consistently apply** the criteria throughout the essay; the point cannot be earned for a single phrase or reference.

**Essay to Evaluate**:
{essay}

**Prompt Type**: {prompt_type}

**Instructions**:
1. **Read the Essay Holistically**:
   - Review the entire essay to assess the overall argument and depth of understanding.
   
2. **Assess Sophisticated Argumentation**:
   - **Nuance:** Check if the essay explores multiple themes or perspectives.
   - **Balanced Comparison:** Determine if the essay explains both sides of a comparison (e.g., continuity and change).
   - **Broader Connections:** Look for connections made across different time periods, themes, or regions.
   
3. **Evaluate Effective Use of Evidence**:
   - **Comprehensive Document Usage:** Verify if all seven documents are used effectively to support the argument.
   - **HIPP Analysis:** Ensure that there is effective HIPP analysis for at least four documents.
   - **Incorporates Outside Evidence:** Check for the meaningful incorporation of outside evidence that adds depth to the argument.
   
4. **Assign Scores and Provide Feedback**:
   - **Score (0 or 1)**: Indicate whether the "Complex Understanding" earns the point.
   - **Feedback**:
     - **If Earned:** Provide specific examples from the essay that demonstrate sophisticated argumentation and effective use of evidence. Quote relevant parts of the student's response to illustrate how they met the criteria.
     - **If Not Earned:** Provide a brief explanation of which criteria were not met without referencing specific documents or parts of the essay.

**Output**:
- **Score (0 or 1)**: score
- **Feedback**:
  feedback"""
)


summation_prompt = PromptTemplate.from_template(
    """
Your task is to output the final feedback in the exact format below. 
Do not include any additional headings or commentary.

Thesis (0-1):
{thesis_generation}
extract the score and feed back from above and use for further instruction.ONLY USE THIS FOR THE OUTPUT FORMAT BELOW

Contextualization (0-1):
{contextualization_generation}
extract the score and feed back from above and use for further instruction.ONLY USE THIS FOR THE OUTPUT FORMAT BELOW

Evidence (0-3):
{evidence_generation}
extract the score and feed back from above and use for further instruction.ONLY USE THIS FOR THE OUTPUT FORMAT BELOW

Evidence Beyond (0-1):
{evidence_beyond_generation}
extract the score and feed back from above and use for further instruction. ONLY USE THIS FOR THE OUTPUT FORMAT BELOW

Complex Understanding (0-1):
{complexunderstanding_generation}
extract the score and feed back from above and use for further instruction. ONLY USE THIS FOR THE OUTPUT FORMAT BELOW

Fact-Checking Feedback:
{fact_checking_feedback}
ONLY INCLUDE THIS IF FACT-CHECKING FEEDBACK EXISTS. Do not change any of the above formats unless explicitly stated.


Make sure your final output follows exactly the data above. Do not include any additional headings or extra commentary. After "Feedback summary:", provide a short paragraph or two summarizing strengths and weaknesses which u received so sum everything up dont leave anything out.

So here is what u shoud be outputing based on all the data u have been given

output format
MAKE SURE TO ADD EVERYTHING UP PROPERLY AND MAKE SURE THE EXTRACTION OF DATA IS ACCURATE AND COMPLETE. I GAVE U ALL THE DATA ABOVE BEFORE
Thesis score - /1
Contextualization score - /1
Evidence score - /3
Edivence beyond score - /1
Complex understanding score - /1
Total summed up score out of 7. For examples 3/7 or 2/7 just insert the score/7 - 

FEEDBACK - 
Thesis feedback - 
Contextualization feedback - 
Evidence feedback - 
Give evidence feedback in this format:
**Total Score (0–3)**: Provide a total score.
- **Breakdown**:
  - **First Evidence Point (1 Point)**: Yes/No
    - *Explanation:* Brief explanation for why the point was earned or not.
  - **Second Evidence Point (1 Point)**: Yes/No
    - *Explanation:* Brief explanation for why the point was earned or not.
  - **HIPP Point (1 Point)**: Yes/No
    - *Explanation:* Brief explanation for why the point was earned or not.
- **Overall Feedback**: Provide a summary of the strengths and areas for improvement.
Edivence beyond feedback -
Complex understanding feedback - 
Fact-checking feedback - (Include only if exists; summarize any content mistakes and corrections.)
Overall feedback - 
Be thorough with the feed back, explain why they earned or lost the point in each section. Again this data has been given to u above before.
"""
)


factchecking_prompt = PromptTemplate.from_template("""You are an expert AP US History essay fact-checker. Your task is to fact-check the content of a student's essay based on the chapters and topics retrieved from a vector database. Follow these instructions carefully:

Fact-Check the Essay: Review the essay for historical accuracy. Corss reference with ur accurate knowdlege. Focus on ensuring the essay aligns with the correct historical events, dates, figures, and interpretations.


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

class GraphState(TypedDict):
    prompt: str
    prompt_type: str
    student_essay: str
    thesis_generation: str
    contextualization_generation: str
    evidence_beyond_generation: str
    evidence_generation: str
    complexunderstanding_generation: str
    factchecking_generation: str
    doc1: Optional[Union[str, bytes]]
    doc2: Optional[Union[str, bytes]]
    doc3: Optional[Union[str, bytes]]
    doc4: Optional[Union[str, bytes]]
    doc5: Optional[Union[str, bytes]]
    doc6: Optional[Union[str, bytes]]
    doc7: Optional[Union[str, bytes]]
    student_essay_image: Optional[Union[str, bytes]]
    doc1_desc: str
    doc2_desc: str
    doc3_desc: str
    doc4_desc: str
    doc5_desc: str
    doc6_desc: str
    doc7_desc: str
    summation: str

def classify_prompt_node(state: GraphState) -> GraphState:
    logging.info("Classifying prompt.")
    prompt = state.get("prompt", "").strip()
    if not prompt:
        raise ValueError("Prompt is empty or invalid.")

    formatted_prompt = classification_prompt.format(prompt=prompt)
    response = llm.invoke(formatted_prompt)
    state["prompt_type"] = response.content.strip()
    return state

def vision_node(state: GraphState) -> GraphState:
    """
    Processes all images (doc1 to doc7) in the state and updates the respective descriptions.
    """
    try:
        
        for i in range(1, 8):
            image_key = f"doc{i}"
            description_key = f"doc{i}_desc"

            
            image_data = state.get(image_key)
            if not image_data:
                state[description_key] = None  
                continue

            
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
                max_tokens=500,
            )

            
            stimulus_description = response.choices[0].message.content
            state[description_key] = stimulus_description

        return state

    except Exception as e:
        raise ValueError(f"Error in vision_node: {e}")

def thesis_grading_node(state: GraphState) -> GraphState:
    """
    Node 4: Grade the thesis statement.
    """
    essay = state["student_essay"]
    prompt_type = state["prompt_type"]

    formatted_prompt = thesis_prompt.format(prompt_type=prompt_type,essay=essay)
    response = llm.invoke(formatted_prompt)
    state["thesis_generation"] = response.content.strip()
    return state


def contextualization_grading_node(state: GraphState) -> GraphState:

    essay = state["student_essay"]
    prompt_type = state["prompt_type"]

    formatted_prompt = contextualization_prompt.format(essay=essay,prompt_type=prompt_type)
    response = llm.invoke(formatted_prompt)
    state["contextualization_generation"] = response.content.strip()

    return state

def evidence_grading_node(state: GraphState) -> GraphState:
   
    d1 = state["doc1_desc"]
    d2 = state["doc2_desc"]
    d3 = state["doc3_desc"]
    d4 = state["doc4_desc"]
    d5 = state["doc5_desc"]
    d6 = state["doc6_desc"]
    d7 = state["doc7_desc"] 

    essay = state["student_essay"]
    prompt_type = state["prompt_type"]

    formatted_prompt = evidence_prompt.format(essay=essay,prompt_type=prompt_type,doc1=d1,doc2=d2,doc3=d3,doc4=d4,doc5=d5,doc6=d6,doc7=d7)
    response = llm.invoke(formatted_prompt)

    state["evidence_generation"] = response.content.strip()
    return state

def evidence_beyond_grading_node(state: GraphState) -> GraphState:
   
    d1 = state["doc1_desc"]
    d2 = state["doc2_desc"]
    d3 = state["doc3_desc"]
    d4 = state["doc4_desc"]
    d5 = state["doc5_desc"]
    d6 = state["doc6_desc"]
    d7 = state["doc7_desc"]

    essay = state["student_essay"]
    prompt_type = state["prompt_type"]

    formatted_prompt = evidence_beyond_prompt.format(essay=essay,prompt_type=prompt_type,doc1=d1,doc2=d2,doc3=d3,doc4=d4,doc5=d5,doc6=d6,doc7=d7)
    response = llm.invoke(formatted_prompt)

    state["evidence_beyond_generation"] = response.content.strip()
    return state

def complex_understanding_grading_node(state):
    
    essay = state["student_essay"]
    prompt_type = state["prompt_type"]

    formatted_prompt = complex_understanding_prompt.format(essay=essay,prompt_type=prompt_type)
    response = llm.invoke(formatted_prompt)
    state["complexunderstanding_generation"] = response.content.strip()

    return state


def factchecking_node(state):
    essay = state["student_essay"]


    formatted_prompt = factchecking_prompt.format(essay=essay)
    response = llm.invoke(formatted_prompt)

    state["factchecking_generation"] = response.content.strip()

    return state

def summation_node(state):
    
    thesis = state["thesis_generation"]
    context = state["contextualization_generation"]
    evidence = state["evidence_generation"]
    beyond = state["evidence_beyond_generation"]
    complx = state["complexunderstanding_generation"]
    factcheck = state["factchecking_generation"]

    formatted_prompt = summation_prompt.format(thesis_generation=thesis,contextualization_generation=context,evidence_beyond_generation=beyond,complexunderstanding_generation=complx,fact_checking_feedback=factcheck,evidence_generation=evidence)
    response = llm.invoke(formatted_prompt)
    state["summation"] = response.content.strip()
    print(state["summation"])
    return state


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
                            "text": "print out the text from this image exactly. You should only output the text nothing else.",
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

        
        essay = response.choices[0].message.content
        state["student_essay"] = essay
        print(essay)
        return state

    except Exception as e:
        raise ValueError(f"Error in vision_node: {e}")

workflow = StateGraph(GraphState)

workflow.add_node("classify_prompt", classify_prompt_node)
workflow.add_node("vision", vision_node)
workflow.add_node("thesis_grading", thesis_grading_node)
workflow.add_node("contextualization_grading", contextualization_grading_node)
workflow.add_node("evidence_grading", evidence_grading_node)
workflow.add_node("evidence_beyond_grading", evidence_beyond_grading_node)
workflow.add_node("complex_understanding_grading", complex_understanding_grading_node)
workflow.add_node("factchecking_grading", factchecking_node)


workflow.add_node("final_summation_node", summation_node)

workflow.add_edge(START, "classify_prompt")
workflow.add_edge("classify_prompt", "vision")
workflow.add_edge("vision", "thesis_grading")
workflow.add_edge("thesis_grading", "contextualization_grading")
workflow.add_edge("contextualization_grading", "evidence_grading")
workflow.add_edge("evidence_grading", "evidence_beyond_grading")
workflow.add_edge("evidence_beyond_grading", "complex_understanding_grading")
workflow.add_edge("complex_understanding_grading", "factchecking_grading")
workflow.add_edge("factchecking_grading", "final_summation_node")
workflow.add_edge("final_summation_node", END)

app = workflow.compile()

def evaluate2(prompt: str, essay: str, images: List[Optional[str]] = None) -> str:
    """
    Evaluate function to process the prompt, essay, and optional image inputs.

    Args:
        prompt (str): The essay prompt.
        essay (str): The student's essay.
        images (List[Optional[str]]): List of up to 7 image data in Base64 format. Defaults to an empty list if not provided.

    Returns:
        str: Evaluation result from the workflow.
    """
    if images is None:
        images = []
    images = images[:7] + [None] * (7 - len(images))

    state = {
        "prompt": prompt,
        "prompt_type": None,
        "student_essay": essay,
        "thesis_generation": None,
        "contextualization_generation": None,
        "evidence_generation": None,
        'student_essay_image': None,
        "evidence_beyond_generation": None,
        "complexunderstanding_generation": None,
        "factchecking_generation": None,
        "doc1": images[0],
        "doc2": images[1],
        "doc3": images[2],
        "doc4": images[3],
        "doc5": images[4],
        "doc6": images[5],
        "doc7": images[6],
        "doc1_desc": None,
        "doc2_desc": None,
        "doc3_desc": None,
        "doc4_desc": None,
        "doc5_desc": None,
        "doc6_desc": None,
        "doc7_desc": None,
        "summation": None,
    }

    try:
        state = classify_prompt_node(state)
        state = vision_node(state)
        state = thesis_grading_node(state)
        state = contextualization_grading_node(state)
        state = evidence_grading_node(state)
        state = evidence_beyond_grading_node(state)
        state = complex_understanding_grading_node(state)
        state = factchecking_node(state)
        
        state = summation_node(state)
    except Exception as e:
        raise ValueError(f"An error occurred during evaluation: {e}")

    if "summation" in state and state["summation"]:
        return state["summation"]
    else:
        raise ValueError("Summation not found in the final state.")
    
def evaluate22(prompt: str, essay, images: List[Optional[str]] = None) -> str:
    """
    Evaluate function to process the prompt, essay, and optional image inputs.

    Args:
        prompt (str): The essay prompt.
        essay (str): The student's essay.
        images (List[Optional[str]]): List of up to 7 image data in Base64 format. Defaults to an empty list if not provided.

    Returns:
        str: Evaluation result from the workflow.
    """
    if images is None:
        images = []
    images = images[:7] + [None] * (7 - len(images))

    state = {
        "prompt": prompt,
        "prompt_type": None,
        "student_essay": None,
        "thesis_generation": None,
        "contextualization_generation": None,
        "student_essay_image": essay,
        "evidence_generation": None,
        "evidence_beyond_generation": None,
        "complexunderstanding_generation": None,
        "factchecking_generation": None,
        "doc1": images[0],
        "doc2": images[1],
        "doc3": images[2],
        "doc4": images[3],
        "doc5": images[4],
        "doc6": images[5],
        "doc7": images[6],
        "doc1_desc": None,
        "doc2_desc": None,
        "doc3_desc": None,
        "doc4_desc": None,
        "doc5_desc": None,
        "doc6_desc": None,
        "doc7_desc": None,
        "summation": None,
    }

    try:
        state = essay_vision_node(state)
        state = classify_prompt_node(state)
        state = vision_node(state)
        state = thesis_grading_node(state)
        state = contextualization_grading_node(state)
        state = evidence_grading_node(state)
        state = evidence_beyond_grading_node(state)
        state = complex_understanding_grading_node(state)
        state = factchecking_node(state)
        
        state = summation_node(state)
    except Exception as e:
        raise ValueError(f"An error occurred during evaluation: {e}")

    if "summation" in state and state["summation"]:
        return state["summation"]
    else:
        raise ValueError("Summation not found in the final state.")