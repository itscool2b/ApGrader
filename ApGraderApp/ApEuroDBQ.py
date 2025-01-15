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

OPENAI_API_KEY = os.getenv('OPENAI_APi_KEY')

openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

index = get_index()

def retriever(query: str, top_k: int = 1) -> List[Dict]:
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
    
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0)



classification_prompt = PromptTemplate.from_template(
    """
        You are a highly accurate and strict teaching assistant for an AP European History class. Your task is to read the DBQ prompt provided by a student and determine which of the three main APUSH DBQ types it falls under:
        
Comparison: The prompt asks the student to compare and/or contrast historical developments, events, policies, or societies.
Causation: The prompt asks the student to explain causes and/or effects of historical events or developments.
Continuity and Change Over Time (CCOT): The prompt asks the student to analyze what changed and what remained the same over a particular time frame.

        Instructions:
        
Read the provided DBQ prompt carefully.
Identify whether the prompt is a Comparison, Causation, or CCOT prompt.
Respond with only one of the three exact words: "Comparison", "Causation", or "CCOT". Do not include any additional text, explanations, or characters. Should be one word

        Student’s Prompt to Classify: {prompt}

        Your Response:
"""
)

thesis_prompt = PromptTemplate.from_template(
    """Evaluate the thesis statement in the following essay based on the provided rubric and evaluation standards:

Rubric for Thesis / Claim:
Refer to the rubric and grade the thesis accordingly based on the rubric.
{rubric}

Evaluation Standards:
Ignore grammar and spelling errors as long as the meaning is clear.
Award 1 point only if the thesis meets all the rubric criteria, but it does not need to be perfectly worded or explictly labled.
Focus on whether the thesis responds to the prompt and establishes a line of reasoning, rather than penalizing for lack of sophistication or minor flaws.
Follow the rubric, be strict but fair. Only award point if the thesis aligns with the rubrics requierments.

Essay to Evaluate:
{essay}

Prompt Type: {prompt_type}

Output:
Score (0 or 1): Indicate whether the thesis earns the point.
Feedback: Provide a brief explanation justifying the score. Highlight which criteria were met or not met.
Be specific: If the thesis earns the point, explain how it meets the rubric criteria. If it does not, explain what is missing or unclear.
"""
)
contextualization_prompt = PromptTemplate.from_template(
    """
Evaluate the contextualization in the following essay based on the provided rubric and evaluation standards:

Rubric for contextulization:
Grade the point based off of this rubric.
{rubric}

Evaluation Standards:
Ignore grammar and spelling errors as long as the meaning is clear.
Award 1 point only if the contextualization meets all rubric criteria.
Do not award partial credit for incomplete or vague contextualization.
Be strict and apply no leniency, if you are not sure if the student deserves the point, do not reward any points.
Contextualization must describe a broader historical event, development, or process relevant to the topic. However, apply leniency when grading for the time frame of the contextualization.
A single phrase or reference does not qualify as contextualization.

Essay to Evaluate:
{essay}



Output:
Score (0 or 1): Indicate whether the contextualization earns the point.
Feedback: Provide a brief explanation justifying the score.
"""
)

evidence_prompt = PromptTemplate.from_template(
    """You are an AP European History (ApEuro) DBQ grader. Your task is to evaluate the Evidence section of a student's essay based on the provided rubric. Follow the instructions carefully to ensure accurate and consistent grading.


Grading Tip: Evaluate whether the student moves beyond simple identification (e.g., "The purpose was to inform…") and provides a meaningful explanation of how the HIPP element enhances their argument in the context of the prompt.

**Evaluation Standards**:
- **Ignore grammar and spelling errors** as long as the meaning is clear.
- **Award points only if the criteria are fully met**; do not award partial credit.
- **Be strict and apply no leniency**; adhere strictly to the rubric.
- **Contextual Analysis (HIPP)** must clearly connect the document features to the argument.
- **Descriptions and usage** of documents must be in the student's own words and relevant to the prompt.

**Document Descriptions and analysis'**:
Here is the descriptions and analysis of each doc.
- **Document 1 Description:** {doc1}
- **Document 2 Description:** {doc2}
- **Document 3 Description:** {doc3}
- **Document 4 Description:** {doc4}
- **Document 5 Description:** {doc5}
- **Document 6 Description:** {doc6}
- **Document 7 Description:** {doc7}
Analyze each document.

**Essay to Evaluate**:
{essay}

**Prompt Type**: {prompt_type}

**Instructions**:
1. **Analyze the Use of Documents**:
    You have been given the analysis of each doc
   - Review each document description (Doc 1 to Doc 7) to determine how accurately and effectively they are described in the essay. 
   - Assess whether the student has explained the content of each document in their own words and related it to the prompt.
   - Identify which documents are used to support the argument and how they contribute to the thesis.
   - If there is any doubt on weather or not the student deserves a point, do not reward point. 
   - Refer back to the documents whenever the student quotes a document/doc to analyze the connections.
   
DBQ Evidence Rubric:
{rubric}

2. **Evaluate Each Evidence Point**:
   - Evaluate and grade the three evidence points based on the DBQ Evidence Rubric given to you above.

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
    """You are an AP European History (ApEuro) DBQ grader. Your task is to evaluate the "Evidence Beyond the Documents" point of a student's essay based on the provided rubric. Follow the instructions carefully to ensure accurate and consistent grading.
    
**Document Descriptions and analysis'**:
Here is the descriptions and analysis of each doc.
- **Document 1 Description:** {doc1}
- **Document 2 Description:** {doc2}
- **Document 3 Description:** {doc3}
- **Document 4 Description:** {doc4}
- **Document 5 Description:** {doc5}
- **Document 6 Description:** {doc6}
- **Document 7 Description:** {doc7}
Analyze each document.

**Analyze the Use of Documents**:
    You have been given the analysis of each doc
   - Review each document description (Doc 1 to Doc 7) to determine how accurately and effectively they are described in the essay. 
   - Assess whether the student has explained the content of each document in their own words and related it to the prompt.
   - Identify which documents are used to support the argument and how they contribute to the thesis.

**DBQ Evidence Beyond the Documents Rubric**:

1. **Requirement:**
   - The student must provide specific evidence that is not in the provided documents. Be sure to look at the analysis of the documents to see if the student is using outside information.
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
    Assign the score based on the rubric for Evidence beyond the documents.
    {rubric}

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
    """You are an AP European History (ApEuro) DBQ grader. Your task is to evaluate the "Complex Understanding" point of a student's essay based on the provided rubric. Follow the instructions carefully to ensure accurate and consistent grading.

**DBQ Complex Understanding Rubric**:
{rubric}

**Evaluation Standards**:
- **Ignore grammar and spelling errors** as long as the meaning is clear.
- **Award 1 point only** if the "Complex Understanding" meets all rubric criteria.
- **Do not award partial credit** for incomplete, vague, or partially correct responses.
- **Be strict but fair and adhere strictly to the rubric.
- **Demonstrate sophistication** through nuanced argumentation and/or effective use of evidence.
- **Consistently apply** the criteria throughout the essay; the point cannot be earned for a single phrase or reference.

**Essay to Evaluate**:
{essay}

**Prompt Type**: {prompt_type}

**Instructions**:
1. **Read the Essay Holistically**:
   - Review the entire essay to assess the overall argument and depth of understanding.
   
2. **Assess Sophisticated Argumentation**:
    use the rubric given above to asses for sophisticted argument.
   
3. **Evaluate Effective Use of Evidence**:
    use the rubric given above to asses effictive use of evidence.
   
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
Total summed up score out of 7. score/7 - 

FEEDBACK - 
Thesis feedback - 
Contextualization feedback - 
Evidence feedback - 
Give evidence feedback in this format:
Total Evidence Score (0–3): Provide a total score.
- Breakdown:
  - First Evidence Point (1 Point): Yes/No
    - Explanation: Brief explanation for why the point was earned or not.
  - Second Evidence Point (1 Point): Yes/No
    - Explanation: Brief explanation for why the point was earned or not.
  - HIPP Point (1 Point): Yes/No
    - Explanation: Brief explanation for why the point was earned or not.
- Overall Feedback: Provide a summary of the strengths and areas for improvement.
Edivence beyond feedback -
Complex understanding feedback - 
Fact-checking feedback - (Include only if exists; summarize any content mistakes and corrections.)
Overall feedback - 
Be thorough with the feed back, explain why they earned or lost the point in each section. Again this data has been given to u above before.

"""
)


factchecking_prompt = PromptTemplate.from_template("""You are an expert AP European History essay fact-checker. Your task is to fact-check the content of a student's essay based on the chapters and topics retrieved from a vector database. Follow these instructions carefully:

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



reflection = PromptTemplate.from_template(
  """
You are an AP Grader tasked with reflecting on your own grading outputs for an AP U.S. History DBQ. Your task is to extract the exact scores from the grading generations provided, ensure adherence to the rubric, and make changes only after thorough review. Your reflection must include:

Rubric:
{rubric}

Prompt Type:
{prompt_type}

Student Essay:
{essay}

Generated Outputs:

Evidence beyond: {beyond}

Thesis Evaluation: {thesis_generation}

Contextualization Evaluation: {contextualization_generation}

Evidence Evaluation: {evidence_generation}

Analysis and Reasoning Evaluation: {complexunderstanding_generation}

Fact-Checking Feedback: {factchecking_generation} (if any)

Your Task:

Extract Scores:

Each generation explicitly contains a score for its respective section. Extract these scores directly as they represent the grading decisions for each part.

Clearly state the points awarded for each section and ensure this information is fully incorporated into the feedback.

Do not interpolate or assume scores—use only the scores explicitly provided in each generation.

Ensure Rubric Adherence:

Carefully review the feedback and scores for each section to ensure alignment with every aspect of the rubric.

Evaluate how well the feedback reflects the rubric for the prompt type (Comparison, Causation, or CCOT).

If changes to the scores are necessary, make them only after thorough review. Clearly explain the reason for any changes.

Feedback Verification and Enhancement:

Check that feedback aligns with the score provided. If feedback contradicts the score, rewrite it using the format: "You put X but Y."

Provide detailed, constructive, and actionable feedback explaining exactly why the student earned or lost points and how they could improve.

For each section, only provide feedback if the student did not earn full points. For sections with full points, do not provide feedback.

For sections where points were lost, follow the format: "You put X and earned Y because Z. However, you could have said W to fully meet the rubric criteria."

Changes and Final Summation:

Specify any changes made to scores, including the original score, the new score, and the reason for the change.

Accurately calculate the total score by summing the scores from each section (after changes, if any).

Output Format:

Section Scores:

Thesis (0-1): Extracted score and explanation

Feedback: Provide feedback only if less than full points were earned.

Contextualization (0-1): Extracted score and explanation

Feedback: Provide feedback only if less than full points were earned.

Evidence (0-2): Extracted score and explanation

First Evidence Point: Yes/No

Feedback: Provide feedback only if the point was not earned.

Second Evidence Point: Yes/No

Feedback: Provide feedback only if the point was not earned.

Analysis and Reasoning (0-2): Extracted score and explanation

Feedback: Provide feedback only if less than full points were earned.

Fact-Checking Feedback: Highlight any factual errors and their impact on scoring, if applicable. Provide constructive corrections in the format:

"You stated X, but the correct information is Y because Z."

Total Score (0-6):

Total Score: Sum the extracted scores explicitly provided in the generations. Reflect any changes here if scores were updated during review.

Changes Made:

Clearly specify any changes to scores, for example:

Thesis: "You put 0 but the thesis meets rubric criteria, so 1 point was awarded."

Evidence: "You put 1 but failed to connect evidence to the argument, so the score was changed to 0."

Final Feedback Summary:

For each section where points were lost, provide feedback in the format: "You put X and earned Y because Z. However, you could have said W to earn the point."

Do not provide feedback for sections where full points were earned.

Be constructive and specific in guiding the student on how to improve.

Conclude with a summary of strengths and areas for improvement based on the rubric. Highlight exactly how the student can improve in future essays.

Example Output:

Section Scores:

Thesis (0-1): 0

Feedback: "You put a vague thesis statement and earned 0 because it lacked a clear argument. However, you could have stated a specific historical argument to fully meet the rubric criteria."

Contextualization (0-1): 1

No feedback necessary.

Evidence (0-2): 1

First Evidence Point: Yes

Second Evidence Point: No

Feedback: "You mentioned evidence but failed to connect it to the thesis. You could have explained how it supports your argument."

Analysis and Reasoning (0-2): 1

Feedback: "You put basic analysis and earned 1 because it addressed the prompt. However, deeper reasoning and exploring counterarguments would have earned 2."

Total Score (0-6): 3/6

Changes Made:

Thesis: "You put 0 but after review, the thesis partially meets criteria, so 1 point was awarded."

Final Feedback Summary:

Strengths: Clear contextualization and use of basic evidence.

Areas for Improvement: Strengthen the thesis with a clearer argument, provide more specific evidence, and deepen analysis by considering different perspectives.

Do not include any extra commentary or user-friendly language. Output the results exactly as specified.


"""
)

def isbs(state):
    prompt = 'If this input is not related to an ap essay at all as in it is something completley not even related to an ap essay and if it is just a few random words or keys then just return the word bs. dont say anything else just bs. If it is related then say not. just the word not. here is the prompt and essay. again dont mistake it for a bad essay that doesnt answer a prompt. that is ok they will just retrive a low score. im saying if they justy spam words or keys or just some random stuff. prompt = {prompt} essay = {essay}. So to summarize if it is completley random then flag it and return the response format whcih i gave. But even if it is remotley related but relally bad it is going to get a low score but dont flag it. Also if its not related to Ap European history flag it. Even if it is an ap essay but its not for euro then flag it. Again if its just a bad prompt os essay leave it be, otherwise if it is completley another topic then flag it.'
    prompt = prompt.format(prompt=state['prompt'],essay=state['student_essay'])
    response = llm.invoke(prompt)
    state['isbsquestion'] = response.content.strip()
    return state

def self_reflection(state):
    thesis = state['thesis_generation']
    contextualization = state['contextualization_generation']
    complex = state['complexunderstanding_generation']
    evidence = state['evidence_generation']
    factcheck = state['factchecking_generation']
    beyond = state['evidence_beyond_generation']
    rubric = state['rubric']
    essay = state['student_essay']
    ptype = state["prompt_type"]
    formatted_prompt = reflection.format(beyond=beyond,prompt_type=ptype,thesis_generation=thesis,contextualization_generation=contextualization,complexunderstanding_generation=complex,evidence_generation=evidence,rubric=rubric,essay=essay,factchecking_generation=factcheck)
    response = llm.invoke(formatted_prompt)
    state['reflection'] = response.content.strip()
    
    return state





class GraphState(TypedDict):
    rubric: List[dict]
    prompt: str
    prompt_type: str
    student_essay: str
    thesis_generation: str
    contextualization_generation: str
    evidence_beyond_generation: str
    evidence_generation: str
    complexunderstanding_generation: str
    factchecking_generation: str
    student_essay_image: Optional[Union[str, bytes]]
    doc1: Optional[Union[str, bytes]]
    doc2: Optional[Union[str, bytes]]
    doc3: Optional[Union[str, bytes]]
    doc4: Optional[Union[str, bytes]]
    doc5: Optional[Union[str, bytes]]
    doc6: Optional[Union[str, bytes]]
    doc7: Optional[Union[str, bytes]]
    doc1_desc: str
    doc2_desc: str
    doc3_desc: str
    doc4_desc: str
    doc5_desc: str
    doc6_desc: str
    doc7_desc: str
    summation: str
    isbsquestion: str
    reflection: str



def retrieve_rubric_node(state):

    query = "ap euro dbq rubric"
    state["rubric"] = retriever(query)

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
                                "text": "Give a full complete description and detailed analysis of the stimulus or image. Be very thorough.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data},
                            },
                        ],
                    }
                ],
                max_tokens=5000,
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
    rubric=state['rubric']
    essay = state["student_essay"]
    prompt_type = state["prompt_type"]

    formatted_prompt = thesis_prompt.format(prompt_type=prompt_type,essay=essay,rubric=rubric)
    response = llm.invoke(formatted_prompt)
    state["thesis_generation"] = response.content.strip()
    return state


def contextualization_grading_node(state: GraphState) -> GraphState:
    rubric=state['rubric']
    essay = state["student_essay"]
    prompt_type = state["prompt_type"]

    formatted_prompt = contextualization_prompt.format(essay=essay,rubric=rubric)
    response = llm.invoke(formatted_prompt)
    state["contextualization_generation"] = response.content.strip()

    return state

def evidence_grading_node(state: GraphState) -> GraphState:
    rubric=state['rubric']
    d1 = state["doc1_desc"]
    d2 = state["doc2_desc"]
    d3 = state["doc3_desc"]
    d4 = state["doc4_desc"]
    d5 = state["doc5_desc"]
    d6 = state["doc6_desc"]
    d7 = state["doc7_desc"] 

    essay = state["student_essay"]
    prompt_type = state["prompt_type"]

    formatted_prompt = evidence_prompt.format(essay=essay,prompt_type=prompt_type,doc1=d1,doc2=d2,doc3=d3,doc4=d4,doc5=d5,doc6=d6,doc7=d7,rubric=rubric)
    response = llm.invoke(formatted_prompt)

    state["evidence_generation"] = response.content.strip()
    return state

def evidence_beyond_grading_node(state: GraphState) -> GraphState:
    rubric=state['rubric']
    d1 = state["doc1_desc"]
    d2 = state["doc2_desc"]
    d3 = state["doc3_desc"]
    d4 = state["doc4_desc"]
    d5 = state["doc5_desc"]
    d6 = state["doc6_desc"]
    d7 = state["doc7_desc"]
    print(d1,d2)
    essay = state["student_essay"]
    prompt_type = state["prompt_type"]

    formatted_prompt = evidence_beyond_prompt.format(essay=essay,prompt_type=prompt_type,doc1=d1,doc2=d2,doc3=d3,doc4=d4,doc5=d5,doc6=d6,doc7=d7,rubric=rubric)
    response = llm.invoke(formatted_prompt)

    state["evidence_beyond_generation"] = response.content.strip()
    return state

def complex_understanding_grading_node(state):
    rubric=state['rubric']
    essay = state["student_essay"]
    prompt_type = state["prompt_type"]

    formatted_prompt = complex_understanding_prompt.format(essay=essay,prompt_type=prompt_type,rubric=rubric)
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
    s = state['student_essay']
    formatted_prompt = summation_prompt.format(thesis_generation=thesis,contextualization_generation=context,evidence_beyond_generation=beyond,complexunderstanding_generation=complx,fact_checking_feedback=factcheck,evidence_generation=evidence)
    response = llm.invoke(formatted_prompt)
    concatenated = '\n\n This is further feedback (in beta) - \n\n'
    state['summation'] = response.content.strip()
    return state['summation'] + concatenated + state['reflection']
    


def evaluateeurodbq(prompt: str, essay: str, images: List[Optional[str]] = None) -> str:
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
        "rubric": [],
        "prompt": prompt,
        "prompt_type": None,
        "student_essay": essay,
        "thesis_generation": None,
        "contextualization_generation": None,
        "evidence_generation": None,
        "student_essay_image": None,
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
        'isbsquestion': None,
        'reflection': None
    }

    try:
        state = isbs(state)
        if state['isbsquestion'] == 'bs':
            return 'give a valid essay pls'
        state = retrieve_rubric_node(state)
        state = classify_prompt_node(state)
        state = vision_node(state)
        state = thesis_grading_node(state)
        state = contextualization_grading_node(state)
        state = evidence_grading_node(state)
        state = evidence_beyond_grading_node(state)
        state = complex_understanding_grading_node(state)
        state = factchecking_node(state)
        
        full_result = summation_node(state)
    except Exception as e:
        raise ValueError(f"An error occurred during evaluation: {e}")

    return full_result
    
def evaluateeurodbqbulk(prompt: str, essay, images: List[Optional[str]] = None) -> str:
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
        'rubric': [],
        "prompt": prompt,
        "prompt_type": None,
        "student_essay": None,
        "student_essay_image": essay,
        "thesis_generation": None,
        "contextualization_generation": None,
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
        'isbsquestion': None,
        'reflection': None
    }

    try:
        state = isbs(state)
        if state['isbsquestion'] == 'bs':
            return 'give a valid essay pls'
        state = retrieve_rubric_node(state)
        state = essay_vision_node(state)
        state = classify_prompt_node(state)
        state = vision_node(state)
        state = thesis_grading_node(state)
        state = contextualization_grading_node(state)
        state = evidence_grading_node(state)
        state = evidence_beyond_grading_node(state)
        state = complex_understanding_grading_node(state)
        state = factchecking_node(state)
        
        full_result = summation_node(state)
    except Exception as e:
        raise ValueError(f"An error occurred during evaluation: {e}")

    return full_result
