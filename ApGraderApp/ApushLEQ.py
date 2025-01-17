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

from typing import List, Dict
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#k
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your environment.")


openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY,max_retries=5)

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



classification_prompt = PromptTemplate.from_template(
    """
You are a highly accurate and strict teaching assistant for an AP U.S. History class. Your task is to read the LEQ prompt provided by a student and determine which of the three main APUSH LEQ types it falls under:
- **Comparison**: The prompt asks the student to compare and/or contrast historical developments, events, policies, or societies.
- **Causation**: The prompt asks the student to explain causes and/or effects of historical events or developments.
- **Continuity and Change Over Time (CCOT)**: The prompt asks the student to analyze what changed and what remained the same over a particular time frame.

**Instructions**:
1. Read the provided LEQ prompt carefully.
2. Identify whether the prompt is a **Comparison**, **Causation**, or **CCOT** prompt.
3. **Respond with only one of the three exact words**: "Comparison", "Causation", or "CCOT". **Do not include any additional text, explanations, or characters. Should be one word**

**Student’s Prompt to Classify**: {prompt}

**Your Response**:
"""
)



thesis_prompt = PromptTemplate.from_template(
    """Evaluate the thesis statement in the following essay based on the provided rubric and evaluation standards:

**Rubric for Thesis / Claim**:
Grade the point based off of this rubric.
{rubric}

**Evaluation Standards**:
- Ignore grammar and spelling errors as long as the meaning is clear.
- Award 1 point only if the thesis meets all rubric criteria.
- Do not award partial credit for incomplete or vague responses.
- Be strict and apply no leniency.

**Essay to Evaluate**:
{essay}

**Prompt Type**: {prompt_type}

**Output**:
- **Score (0 or 1)**: Indicate whether the thesis earns the point.
- **Feedback**: Provide a brief explanation justifying the score.
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
Be strict and apply no leniency.
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
    """Evaluate the evidence and support for the argument in the following essay based on the rubric and evaluation standards:

Rubric for Evidence & Support for Argument:
Grade the point based on this rubric.
{rubric}

Evaluation Standards:
- Ignore grammar and spelling errors as long as the meaning is clear.
- Award the first point (Evidence Usage) **only** if at least two specific historical examples are identified and relevant to the prompt.
- Award the second point (Evidence Supporting Argument) **only** if:
  - The first point (Evidence Usage) has been awarded.
  - The response uses at least two specific and relevant pieces of historical evidence to support an argument, it should connect to the thesis.**
  - The evidence provided is **historically accurate** and falls within the time frame and context of the prompt.
  - The evidence supports the argument that the student is making in a logical way.
- Be strict and apply no leniency. But be fair, recognize attempts of reasoning. Do not award partial credit for incomplete, vague, irrelevant, or inaccurate responses.

Evaluation Process:
1. Analyze the essay holistically and identify the thesis or argument being made.
2. Determine whether the response provides specific and historically accurate evidence relevant to the time frame and context of the prompt.
3. Evaluate whether the identified evidence is clearly and explicitly tied to the thesis through logic.
4. Ensure feedback includes specific references to the essay content and explains why points were awarded or not.

Essay to Evaluate:
{essay}

Prompt Type: {prompt_type}

Output:
Total Score (0–2): Provide a total score.
Breakdown:
- First Evidence Point (1 Point): Yes/No
  - Explanation: Brief explanation for why the point was earned or not.
- Second Evidence Point (1 Point): Yes/No
  - Explanation: Brief explanation for why the point was earned or not.
Feedback: Provide a detailed explanation for the score, including:
- Specific examples of evidence identified from the essay.
- An assessment of how well the evidence supports the argument, emphasizing cause-and-effect relationships and historical accuracy.
- Reasons for any points not awarded, referencing specific weaknesses or gaps in the essay's use of evidence and support.
"""
)
complexunderstanding_prompt = PromptTemplate.from_template(
    """Evaluate the Analysis and Reasoning in the following essay based on the provided rubric and evaluation standards. The evaluation must consider the type of prompt (Comparison, Causation, or CCOT) when awarding points for historical reasoning.

**Rubric for Analysis and Reasoning**:
Grade the points based on this rubric.
{rubric}

**Evaluation Standards**:
- **Grammar and Spelling**: Ignore any errors as long as the meaning is clear.
- **Historical Reasoning**:
  - **Comparison**: Use comparative analysis to highlight similarities and differences between two or more historical developments.
  - **Causation**: Explain cause-and-effect relationships, identifying multiple causes or effects as relevant to the prompt.
  - **CCOT (Continuity and Change Over Time)**: Analyze both continuities and changes, explaining how and why specific elements persisted or transformed over the specified period.
- **Award Historical Reasoning Points**:
  - **0 Points**: No use of appropriate historical reasoning.
  - **1 Point**: Demonstrates the use of historical reasoning aligned with the prompt type, showing clear understanding.
  - **2 Points**: Effectively uses historical reasoning aligned with the prompt type, demonstrating thorough and insightful understanding.
- **Complex Understanding**:
  - **Award Complex Understanding Points Only If**:
    1. The Historical Reasoning point has been awarded (either 1 or 2).
    2. The essay demonstrates meaningful argumentation and/or effective use of evidence as outlined in the rubric.
  - **Criteria for Complex Understanding**:
    - **Meaningful Argumentation**:
      - Explores multiple themes or perspectives.
      - Analyzes similarities and differences, continuities and changes, or multiple causes and effects.
      - Makes connections within and across periods or geographical areas.
      - Considers different viewpoints or counterarguments.
    - **Effective Use of Evidence**:
      - Uses at least two specific and relevant pieces of evidence.
      - Integrates evidence to support a coherent argument.
      - Demonstrates an understanding of different perspectives relevant to the prompt.
  - **Scoring for Complex Understanding**:
    - **0 Points**: Does not demonstrate meaningful argumentation or effective use of evidence.
    - **1 Point**: Demonstrates some aspects of meaningful argumentation and/or effective use of evidence but lacks full development.
    - **2 Points**: Clearly demonstrates meaningful argumentation and/or effective use of evidence, meeting the criteria effectively.
- **Flexibility in Scoring**:
  - Apply the rubric with an understanding of varying essay strengths.
  - Allow partial fulfillment of criteria to recognize diverse ways students may demonstrate understanding.
  - Award points based on the presence of key elements without requiring every aspect to be fully developed.

**Essay to Evaluate**:
{essay}

**Prompt Type**: {prompt_type}

**Output**:
- **Score (0, 1, or 2)**: Indicate the total points awarded for the Analysis and Reasoning section.
- **Feedback**: Provide a brief explanation for the score, including:
  - **Historical Reasoning**: How the response used reasoning aligned with the specific type of prompt (e.g., comparison, causation, CCOT).
  - **Complex Understanding**: How the response demonstrated a meaningful or effective argument, referencing specific elements of the essay.
  - **Reasons for Points Not Awarded**: Clearly state why certain points were not awarded based on the rubric criteria.
"""
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

Evidence (0-2):
{evidence_generation}
extract the score and feed back from above and use for further instruction.ONLY USE THIS FOR THE OUTPUT FORMAT BELOW

Analysis and Reasoning (0-2):
{complexunderstanding_generation}
extract the score and feed back from above and use for further instruction. ONLY USE THIS FOR THE OUTPUT FORMAT BELOW

Fact-Checking Feedback:
{fact_checking_feedback}
ONLY INCLUDE THIS IF FACT-CHECKING FEEDBACK EXISTS. Do not change any of the above formats unless explicitly stated.


Make sure your final output follows exactly the data above. Do not include any additional headings or extra commentary. After "Feedback summary:", provide a short paragraph or two summarizing strengths and weaknesses which u received so sum everything up dont leave anything out.

So here is what u shoud be outputing based on all the data u have been given

output format
MAKE SURE TO ADD EVERYTHING UP PROPERLY AND MAKE SURE THE EXTRACTION OF DATA IS ACCURATE AND COMPLETE. I GAVE U ALL THE DATA ABOVE BEFORE
Thesis score - 
contextualization score - 
evidence score - 
complex understanding score - 
total summed up score out of 6. For examples 3/6 or 2/6 just insert the score/6 - 

FEEDBACK - 
Thesis feedback - 
contextualization feedback - 
evidence feedback - 
Total Score (0–2): Provide a total score.
Breakdown:
First Evidence Point (1 Point): Yes/No
Explanation: Brief explanation for why the point was earned or not.
Second Evidence Point (1 Point): Yes/No
Explanation: Brief explanation for why the point was earned or not.
Feedback: Provide a brief explanation for the score, including:
Specific examples of evidence identified.
An assessment of how well the evidence supports the argument.
Reasons for any points not awarded.
complex understanding feedback - 
fact-checking feedback - (Include only if exists; summarize any content mistakes and corrections.)
overall feedback - 
Be thorough with the feed back, explain why they earned or lost the point in each section. Again this data has been given to u above before.


"""
)
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


reflection = PromptTemplate.from_template(
  """
You are an AP Grader tasked with reflecting on your own grading outputs for an AP U.S. History LEQ. Your task is to extract the exact scores from the grading generations provided, ensure adherence to the rubric, and make changes only after thorough review. Your reflection must include:

Rubric:
{rubric}

Prompt Type:
{prompt_type}

Student Essay:
{essay}

Generated Outputs:

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



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0)


from typing import List, Dict, Optional, Union, TypedDict, Any
class GraphState(TypedDict):
    """
    Represents the state of the graph workflow.
    """
    prompt: str
    prompt_type: str
    student_essay: str
    relevant_essays: List[Dict]
    rubric: List[Dict]  
    thesis_generation: str
    relevant_chapters: List[Dict]
    contextualization_generation: str
    evidence_generation: str
    complexunderstanding_generation: str
    factchecking_generation: str
    summation: str
    summation: str
    isbsquestion: str
    student_essay_image: Optional[Union[str, bytes]]

workflow = StateGraph(GraphState)


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
    
    
    formatted_prompt = prompt.format(prompt=state['prompt'], essay=state['student_essay'])
    
    
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


def self_reflection(state):
    thesis = state['thesis_generation']
    contextualization = state['contextualization_generation']
    complex = state['complexunderstanding_generation']
    evidence = state['evidence_generation']
    factcheck = state['factchecking_generation']
    rubric = state['rubric']
    essay = state['student_essay']
    ptype = state["prompt_type"]
    formatted_prompt = reflection.format(prompt_type=ptype,thesis_generation=thesis,contextualization_generation=contextualization,complexunderstanding_generation=complex,evidence_generation=evidence,rubric=rubric,essay=essay,factchecking_generation=factcheck)
    response = llm.invoke(formatted_prompt)
    state['reflection'] = response.content.strip()
    
    return state

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
    

def classify_prompt_node(state: GraphState) -> GraphState:
    logging.info("Classifying prompt.")
    prompt = state.get("prompt", "").strip()
    if not prompt:
        raise ValueError("Prompt is empty or invalid.")

    formatted_prompt = classification_prompt.format(prompt=prompt)
    response = llm.invoke(formatted_prompt)
    state["prompt_type"] = response.content.strip()
    return state
   

def retrieve_essays_node(state):

    prompt_type = state["prompt_type"]
    query = f"all essays of this prompt type {prompt_type}"
    state["relevant_essays"] = retriever(query)

    return state

    

def fetch_rubric_node(state: GraphState) -> GraphState:
    """
    Node to fetch rubric documents using the retriever function.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        GraphState: Updated state with fetched rubric documents.
    """
    try:
        logging.info("Fetching rubric documents.")
        state["rubric"] = retriever("ap us history leq rubric")
        logging.info(f"Fetched {len(state['rubric'])} rubric documents.")
    except Exception as e:
        logging.error(f"Error in fetch_rubric_node: {e}")
        raise RuntimeError(f"Error in fetch_rubric_node: {e}")
    return state


ch_prompt = PromptTemplate.from_template("""

This is the student essay - {essay}

Write a query that i could put in a vector db to find relevant chapters to fact check the content of the essay. I already have anoth4r prompt to fact check and i also pass in chapters.

So here should be your output

**ouput**

A thorough query to find relevant chpaters based off the student essay to fact check. Your output should only consist of the query, that is it. thats it

""")

def retrieve_chapters_node(state: GraphState) -> GraphState:
    essay = state["student_essay"]

    
    formatted_prompt = ch_prompt.format(essay=essay)
    response = llm.invoke(formatted_prompt)

    query = response.content.strip()
    state["relevant_chapters"] = retriever(query)

    return state



def thesis_grading_node(state: GraphState) -> GraphState:
    """
    Node 4: Grade the thesis statement.
    """
    rubric = state["rubric"]
    essay = state["student_essay"]
    prompt_type = state["prompt_type"]

    formatted_prompt = thesis_prompt.format(rubric=rubric,prompt_type=prompt_type,essay=essay)
    response = llm.invoke(formatted_prompt)
    state["thesis_generation"] = response.content.strip()
    return state


def contextualization_grading_node(state: GraphState) -> GraphState:
    rubric = state["rubric"]
    essay = state["student_essay"]
    prompt_type = state["prompt_type"]

    formatted_prompt = contextualization_prompt.format(rubric=rubric,essay=essay)
    response = llm.invoke(formatted_prompt)
    state["contextualization_generation"] = response.content.strip()

    return state

def evidence_grading_node(state: GraphState) -> GraphState:
   
    rubric = state["rubric"]
    essay = state["student_essay"]
    prompt_type = state["prompt_type"]

    formatted_prompt = evidence_prompt.format(rubric=rubric,essay=essay,prompt_type=prompt_type)
    response = llm.invoke(formatted_prompt)

    state["evidence_generation"] = response.content.strip()
    return state


def analysis_grading_node(state: GraphState) -> GraphState:
    rubric = state["rubric"]
    essay = state["student_essay"]
    prompt_type = state["prompt_type"]
      
    formatted_prompt = complexunderstanding_prompt.format(rubric=rubric,essay=essay,prompt_type=prompt_type)
    response = llm.invoke(formatted_prompt)

    state["complexunderstanding_generation"] = response.content.strip()

    return state


def fact_check_node(state):
    
    essay = state["student_essay"]
    chapters = state["relevant_chapters"]

    formatted_prompt = factchecking_prompt.format(essay=essay,chapters=chapters)
    response = llm.invoke(formatted_prompt)

    state["factchecking_generation"] = response.content.strip()

    return state

def final_node(state: dict) -> dict:
    """
    Final node to compute the summation and update the state.
    """
    try:
        
        thesis = state["thesis_generation"]
        cont = state["contextualization_generation"]
        evidence = state["evidence_generation"]
        complexu = state["complexunderstanding_generation"]
        ptype = state["prompt_type"]
        fact = state["factchecking_generation"]
        student_essay = state['student_essay']
        formatted_prompt = summation_prompt.format(
            thesis_generation=thesis,
            contextualization_generation=cont,
            evidence_generation=evidence,
            complexunderstanding_generation=complexu,
            fact_checking_feedback=fact,
            student_essay=student_essay
        )

        concatenated = '\n\n This is more feedback (beta) - \n\n'
        response = llm.invoke(formatted_prompt)
        #ds
        state['summation'] = response.content.strip()
        if state.get('student_essay_image') is not None:
            final_output = f"{state['summation']}{concatenated}{state['reflection']}\n\n This the essay we extracted. if it is not accurate or if it is missing words, our apologies. We suggest you type in the missing partts and submit through the text submission for the best accuracy. {state['student_essay']}"
            return final_output
        final_output = f"{state['summation']}{concatenated}{state['reflection']}"
        return final_output
    except Exception as e:
        raise RuntimeError(f"Error in final_node: {e}")








workflow.add_node("classify_prompt", classify_prompt_node)
workflow.add_node("retrieve_essays", retrieve_essays_node)
workflow.add_node("fetch_rubric", fetch_rubric_node)
workflow.add_node("retrieve_chapters", retrieve_chapters_node)
workflow.add_node("thesis_grading", thesis_grading_node)
workflow.add_node("contextualization_grading", contextualization_grading_node)
workflow.add_node("evidence_grading", evidence_grading_node)
workflow.add_node("analysis_grading", analysis_grading_node)
workflow.add_node("factchecking_grading", fact_check_node)
workflow.add_node("final_node", final_node)


workflow.add_edge(START, "classify_prompt")  
workflow.add_edge("classify_prompt", "retrieve_essays")  
workflow.add_edge("retrieve_essays", "fetch_rubric")  
workflow.add_edge("fetch_rubric", "retrieve_chapters")  
workflow.add_edge("retrieve_chapters", "thesis_grading")  
workflow.add_edge("thesis_grading", "contextualization_grading")  
workflow.add_edge("contextualization_grading", "evidence_grading")  
workflow.add_edge("evidence_grading", "analysis_grading")  
workflow.add_edge("analysis_grading", "factchecking_grading")  
workflow.add_edge("factchecking_grading", "final_node")  
workflow.add_edge("final_node", END)  



app = workflow.compile()


def evaluate(prompt: str, essay: str) -> str:
    
    state = {
        "prompt": prompt,
        "prompt_type": None,
        "student_essay": essay,
        "rubric": [],
        "relevant_essays": [],
        "relevant_chapters": [],
        "thesis_generation": None,
        "contextualization_generation": None,
        "evidence_generation": None,
        "complexunderstanding_generation": None,
        "factchecking_generation": None,
        "summation": None,
        'reflection': None,
        'isbsquestion': None,
    }

    
    state = classify_prompt_node(state)  
    state = retrieve_essays_node(state)  
    state = fetch_rubric_node(state)  
    state = retrieve_chapters_node(state)  
    state = thesis_grading_node(state)  
    state = contextualization_grading_node(state)  
    state = evidence_grading_node(state)  
    state = analysis_grading_node(state)  
    state = fact_check_node(state)  
    state = self_reflection(state)
    state = final_node(state)  

    
    return state

def evaluate69(prompt: str, essay) -> str:
    
    state = {
        "prompt": prompt,
        "prompt_type": None,
        "student_essay": None,
        "student_essay_image": essay,
        "rubric": [],
        "relevant_essays": [],
        "relevant_chapters": [],
        "thesis_generation": None,
        "contextualization_generation": None,
        "evidence_generation": None,
        "complexunderstanding_generation": None,
        "factchecking_generation": None,
        "summation": None,
        'reflection': None,
        'isbsquestion': None,
    }
    state = essay_vision_node(state)
    if state['student_essay'] == None:
            return 'we tried mutliple times to recognize your essay. You can try resubmitting but we have tried out best. Sorry'
    

    
    state = classify_prompt_node(state)  
    state = retrieve_essays_node(state)  
    state = fetch_rubric_node(state)  
    state = retrieve_chapters_node(state)  
    state = thesis_grading_node(state)  
    state = contextualization_grading_node(state)  
    state = evidence_grading_node(state)  
    state = analysis_grading_node(state)  
    state = fact_check_node(state)  
    state = self_reflection(state)
    state = final_node(state)  

    
    return state
