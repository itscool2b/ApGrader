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
- Responds to the prompt with a historically defensible thesis or claim.
- Establishes a line of reasoning.
- Makes a claim that responds to the prompt (not merely restating or rephrasing it).
- Consists of one or more sentences located in one place, either in the introduction or conclusion.
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

**Rubric for Contextualization**:
- Describes a broader historical context relevant to the prompt.
- Relates the topic to broader historical events, developments, or processes before, during, or after the time frame of the question.
- Not awarded for merely a phrase or reference.
{rubric}

**Evaluation Standards**:
- Ignore grammar and spelling errors as long as the meaning is clear.
- Award 1 point only if the contextualization meets all rubric criteria.
- Do not award partial credit for incomplete or vague contextualization.
- Be strict and apply no leniency.
- Contextualization must describe a broader historical event, development, or process relevant to the topic.
- A single phrase or reference does not qualify as contextualization.

**Essay to Evaluate**:
{essay}

**Prompt Type**: {prompt_type}

**Output**:
- **Score (0 or 1)**: Indicate whether the contextualization earns the point.
- **Feedback**: Provide a brief explanation justifying the score.
"""
)

evidence_prompt = PromptTemplate.from_template(
    """Evaluate the evidence and support for the argument in the following essay based on the rubric and evaluation standards:

**Rubric for Evidence & Support for Argument**:

1. **Evidence Usage (1 point)**:
   - The response must provide at least two specific historical examples relevant to the topic of the prompt.
   - Examples must be concrete and historically accurate, such as specific events, dates, policies, or figures.
   - Vague references or generalizations do not satisfy this requirement.

2. **Evidence Supporting Argument (1 point)**:
   - The response must use at least two specific and relevant pieces of historical evidence to support an argument in response to the prompt.
   - Evidence must be integrated into the argument, demonstrating how it supports the thesis.
   - Analysis and explanation are required to establish a clear connection between the evidence and the thesis.
   - Simply listing evidence without analysis does not meet this criterion.
{rubric}

**Evaluation Standards**:
- Ignore grammar and spelling errors as long as the meaning is clear.
- Award the first point (**Evidence Usage**) only if at least two specific historical examples are identified and relevant.
- Award the second point (**Evidence Supporting Argument**) only if:
  1. The first point (**Evidence Usage**) has been awarded.
  2. The response uses at least two specific and relevant pieces of historical evidence to support an argument, with clear and logical connections to the thesis.
- Be strict and apply no leniency.
- Do not award partial credit for incomplete, vague, or irrelevant responses.
- Ensure that all evidence is historically accurate and directly supports the argument.

**Essay to Evaluate**:
{essay}

**Prompt Type**: {prompt_type}

**Output**:
- **Score (0, 1, or 2)**: Indicate the total points awarded for the Evidence & Support for Argument section.
- **Feedback**: Provide a brief explanation for the score, including:
  - Specific examples of evidence identified.
  - An assessment of how well the evidence supports the argument.
  - Reasons for any points not awarded.
"""
)

complexunderstanding_prompt = PromptTemplate.from_template(
    """Evaluate the analysis and reasoning in the following essay based on the rubric and evaluation standards. The evaluation must take into account the type of prompt (Comparison, Causation, or CCOT) when awarding points for historical reasoning.

**Rubric for Analysis and Reasoning**:

1. **Historical Reasoning (1 point)**:
   - The response must use historical reasoning (e.g., comparison, causation, continuity and change over time) to frame or structure an argument that addresses the prompt.
   - To earn this point:
     - The reasoning should align with the specific type of prompt provided (e.g., comparison must compare and contrast, causation must address causes or effects, CCOT must highlight changes and continuities).
     - The reasoning may be uneven, imbalanced, or general, but it must clearly demonstrate an attempt to structure the argument around historical reasoning.

2. **Complex Understanding (1 point)**:
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

**Evaluation Standards**:
- Ignore grammar and spelling errors as long as the meaning is clear.
- Award the Historical Reasoning point only if the response uses reasoning aligned with the specific type of prompt (**Comparison**, **Causation**, **CCOT**).
- Award the Complex Understanding point only if:
  1. The Historical Reasoning point has been awarded.
  2. The essay demonstrates sophisticated argumentation and/or nuanced use of evidence as outlined in the rubric.
- Be strict and apply no leniency. Do not award partial credit for incomplete, vague, or irrelevant responses.

**Essay to Evaluate**:
{essay}

**Prompt Type**: {prompt_type}

**Output**:
- **Score (0, 1, or 2)**: Indicate the total points awarded for the Analysis and Reasoning section.
- **Feedback**: Provide a brief explanation for the score, including:
  - For Historical Reasoning: How the response used reasoning aligned with the specific type of prompt (e.g., comparison, causation, CCOT).
  - For Complex Understanding: How the response demonstrated a nuanced or sophisticated argument, referencing specific elements of the essay.
  - Reasons for any points not awarded.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")


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
    
    student_essay_image: Optional[Union[str, bytes]]

workflow = StateGraph(GraphState)


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
                            "text": "print out the text from this image exactly. You should only output the text nothing else. make sure it is complete and accurate.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data},  
                        },
                    ],
                }
            ],
            max_tokens=10000,
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

    formatted_prompt = contextualization_prompt.format(rubric=rubric,essay=essay,prompt_type=prompt_type)
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
        
        formatted_prompt = summation_prompt.format(
            thesis_generation=thesis,
            contextualization_generation=cont,
            evidence_generation=evidence,
            complexunderstanding_generation=complexu,
            fact_checking_feedback=fact,
        )

        
        response = llm.invoke(formatted_prompt)

        
        if hasattr(response, "content") and response.content.strip():
            state["summation"] = response.content.strip()
        else:
            raise ValueError("Summation generation failed.")

        return state

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
    state = final_node(state)  

    
    if "summation" in state and state["summation"]:
        return state["summation"]
    else:
        raise ValueError("Summation not found in the final state.")

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
    }

    state = essay_vision_node(state)
    state = classify_prompt_node(state)  
    state = retrieve_essays_node(state)  
    state = fetch_rubric_node(state)  
    state = retrieve_chapters_node(state)  
    state = thesis_grading_node(state)  
    state = contextualization_grading_node(state)  
    state = evidence_grading_node(state)  
    state = analysis_grading_node(state)  
    state = fact_check_node(state)  
    state = final_node(state)  

    
    if "summation" in state and state["summation"]:
        return state["summation"]
    else:
        raise ValueError("Summation not found in the final state.")
