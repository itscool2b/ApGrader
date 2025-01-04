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

case1 = PromptTemplate.from_template("""
You are an expert AP U.S. History SAQ grader. Your task is to evaluate the provided essay response for a specific SAQ question. Each question may contain multiple subparts (e.g., A, B, C), and you must analyze and grade each subpart individually based on the provided rubric.

**IMPORTANT NOTE**: 
- Do not award a 0 unless the essay is entirely unrelated or lacks any substantive content.
- Award partial credit based on the quality and accuracy of each subpart's response.

### **Grading Process:**

#### **1. Read the SAQ Question and Student Response:**

- **Question:**
  {questions}

- **Student’s Response:**
  {essay}

#### **2. Analyze the Structure:**

- **Identify Subparts:**
  - Determine how many subparts (A, B, C, etc.) the question contains.
  - Understand the specific task each subpart is asking for based on the task verbs (e.g., Identify, Describe, Explain).

#### **3. Grade Each Subpart (A, B, C, etc.):**

- **General Criteria for All Subparts:**
  - **Identify:** Correctly name or state a relevant historical fact.
  - **Describe:** Provide specific details about a historical fact, event, or concept beyond mere terminology.
  - **Explain:** Analyze cause/effect, context, or significance with clear connections.
  - **Accuracy:** Ensure all content is historically defensible and accurate.
  - **Relevance:** Ensure responses directly address the prompt and are pertinent to the historians' arguments or the question's requirements.

- **Specific Criteria Based on Task Verbs:**
  
  - **For "Identify":**
    - **Accuracy:** Correctly names a relevant historical fact or figure.
    - **Relevance:** The identified fact directly pertains to the question.
  
  - **For "Describe":**
    - **Specificity:** Provides detailed information about the event or concept.
    - **Depth:** Goes beyond surface-level details to convey a deeper understanding.
  
  - **For "Explain":**
    - **Cause/Effect:** Clearly articulates the relationship between events or concepts.
    - **Context:** Demonstrates an understanding of the broader historical context.
    - **Significance:** Highlights the importance or impact of the event or concept.
  
  - **For "Compare":**
    - **Similarity/Difference:** Clearly discusses both similarities and/or differences.
    - **Connection:** Directly relates the comparison to the question's focus.

#### **4. Provide Feedback for Each Subpart:**

- **If the response earns the point:**
  - **Explain why it is correct**, referencing specific aspects of the response that meet the criteria.
  - **Highlight the alignment** between the student’s explanation and the expected historical interpretation or analysis.

- **If the response does not earn the point:**
  - **Explain what is missing, incorrect, or vague**.
  - **Provide constructive guidance** on how to improve the response, focusing on accuracy, relevance, and depth of analysis.

#### **5. Summarize the Total Score:**

- **Total Score:** Sum the points earned across all subparts (e.g., 0–3 for a 3-part SAQ).
- **Strengths:** Highlight specific aspects of the student's responses that were well-executed.
- **Areas for Improvement:** Suggest targeted ways to enhance their responses, focusing on accuracy, relevance, and depth.
- **Originality Note:** *(Only if applicable)* Comment on the originality and appropriateness of supporting evidence used, ensuring they reference events or concepts not superficially mentioned.

### **Grading Criteria:**

- **Accuracy:**
  - Responses must be historically defensible and free from significant factual errors.
  
- **Relevance:**
  - Responses must directly address the question and remain focused on the task.
  
- **Depth:**
  - Responses should demonstrate a thorough understanding of the historical concepts, events, or figures involved.
  
- **Specificity:**
  - Vague or generalized answers will not earn full credit. Specific details and examples are required.
  
- **Clarity:**
  - Responses should be clearly written and logically organized, making the argument easy to follow.

### **Output Format for Grading:**

- **Feedback for Each Subpart:**

  - **A: (Score: X/1)**  
    *Feedback:* [Detailed feedback...]

  - **B: (Score: X/1)**  
    *Feedback:* [Detailed feedback...]

  - **C: (Score: X/1)**  
    *Feedback:* [Detailed feedback...]

- **Total Score and Summary:**

  - **Total Score:** X/Y (e.g., 2/3 for a 3-part question).
  - **Strengths:** Highlight what the student did well, referencing specific parts of their responses.
  - **Areas for Improvement:** Suggest specific ways to improve their response, focusing on accuracy, relevance, and depth.

### **Additional Instructions:**

- **Precision Over Presence:** Ensure that points are awarded based on the **quality and accuracy** of the response rather than merely the presence of relevant keywords or events.
- **Alignment with Historians' Arguments:** Pay close attention to whether the student's explanations accurately reflect the specific arguments and interpretations expected by the question.
- **Avoid Over-Awarding Points:** Be stringent in evaluating the depth and correctness of the connections made by the student, ensuring that partial or incorrect answers do not earn full credit.
- **Constructive Feedback:** Provide feedback that not only points out deficiencies but also guides the student on how to address them, enhancing their historical analysis skills.
                                     VERY IMPORTANT!! U CANNOT GIVE A 0.5. only a 1 or 0 for each section   VERY IMPORTANT. DONT BE LENIANT. if it fits the rubric give the point. otherwise be less leniant. """)

case2 = PromptTemplate.from_template("""
You are an expert AP U.S. History SAQ grader. This question includes a stimulus (image, chart, or visual resource) that must be analyzed and integrated into the evaluation process. Your task is to grade each subpart of the SAQ (A, B, C, etc.) while taking the provided stimulus into account.

**IMPORTANT NOTE**: 
- Do not award a 0 unless the essay is entirely unrelated or lacks any substantive content.
- **ONLY** award a 1 or 0 for each subpart. Do not give partial credit (e.g., 0.5).
- Be **extremely stringent** in grading to ensure alignment with human AP History teachers' standards.

### **Grading Process:**

#### **1. Analyze the Provided Stimulus:**

- **Stimulus Description:**
  {stimulus}

- **Actions:**
  - Carefully review the stimulus and identify its key details (e.g., captions, labels, or visual elements).
  - Understand how the stimulus relates to the historical context and themes referenced in the question.
  - Determine if and how the stimulus should be integrated into the student's response based on the question's requirements.

#### **2. Read the SAQ Question and Student Response:**

- **Question:**
  {questions}

- **Student’s Response:**
  {essay}

#### **3. Grade Each Subpart (A, B, C, etc.):**

- **General Criteria for All Subparts:**
  - **Identify:** Correctly name or state a relevant historical fact.
  - **Describe:** Provide specific details about a historical event or concept beyond mere terminology.
  - **Explain:** Analyze cause/effect, context, or significance with clear connections.
  - **Accuracy:** Ensure all content is historically defensible and accurate.
  - **Relevance:** Ensure responses directly address the prompt and are pertinent to the historians' arguments.
  - **Clarity and Depth:** Responses must be clear, well-organized, and demonstrate a thorough understanding of the topic.

- **Specific Criteria Based on Subpart Requirements:**

  - **For Subparts Requiring Stimulus Analysis (e.g., Part A):**
    - **Accuracy:**
      - Does the response correctly reference details or themes from the stimulus?
      - Does the interpretation of the stimulus align with its intended historical context?
    - **Relevance:**
      - Is the stimulus used effectively to address the question?
      - Does the response integrate the stimulus seamlessly with broader historical knowledge?
    - **Depth of Understanding:**
      - Does the response demonstrate a nuanced understanding of how the stimulus relates to the historians' arguments?
  
  - **For Subparts Requiring Historical Events Not in Stimulus (e.g., Parts B and C):**
    - **Originality:**
      - Is the historical event or development referenced **not directly mentioned** in the provided excerpts?
    - **Timeframe Adherence:**
      - Does the referenced event fall within the specified timeframe (e.g., 1600-1700)?
    - **Accuracy:**
      - Is the referenced event historically accurate within the specified timeframe?
      - Does the event accurately reflect the historians' interpretations?
    - **Relevance and Support:**
      - Does the event effectively support the specific historian’s argument as required by the question?
      - Is the connection between the event and the historian’s argument clearly articulated and logically sound?
    - **Depth of Explanation:**
      - Does the response provide a thorough and detailed explanation of how the evidence supports the argument?
      - Avoids vague or general statements; includes specific details and clear connections.

#### **4. Provide Feedback for Each Subpart:**

- **If the response earns the point:**
  - **Explain why it is correct**, including specific references to the stimulus if required.
  - **Highlight the alignment** between the student’s explanation and the historians' arguments.

- **If the response does not earn the point:**
  - **Explain what is missing, incorrect, or vague**.
  - **For Parts B and C:**
    - Specifically address if the historical event used is from the stimulus and instruct to use events not covered in the excerpts.
    - **Emphasize Timeframe Adherence:** Point out if the evidence falls outside the specified period.
  - **Provide constructive guidance** on how to improve the response, focusing on accuracy, relevance, and depth of analysis.

#### **5. Summarize the Total Score:**

- **Total Score:** Sum the points earned across all subparts (e.g., 0–3 for a 3-part SAQ).
- **Strengths:** Highlight specific aspects of the student's responses that were well-executed, referencing specific parts of their responses.
- **Areas for Improvement:** Suggest targeted ways to enhance their responses, focusing on accuracy, relevance, and depth of analysis.
- **Originality Note:** Comment on the originality and appropriateness of the supporting evidence used in Parts B and C, ensuring they reference events not included in the excerpts.

### **Grading Criteria:**

- **For Subparts Requiring the Stimulus:**
    DO NOT APPLY ANY LENIENCY IN GRADING. BE STRICT but fair. 
  - **Accuracy:**
    - Correctly references details or themes from the stimulus.
  - **Relevance:**
    - Uses the stimulus to directly address the question.
  - **Integration with Broader Knowledge:**
    - Connects observations from the stimulus to broader historical themes or developments with depth and precision.

- **For Subparts Requiring Historical Events Not in Stimulus (e.g., Parts B and C):**
  - **Originality:**
    - References historical events or developments **not mentioned** in the provided excerpts.
  - **Timeframe Adherence:**
    - References fall within the specified timeframe (e.g., 1600-1700).
  - **Accuracy:**
    - References are historically accurate and fall within the specified timeframe.
    - Reflect the historians' interpretations accurately.
  - **Relevance and Support:**
    - Events effectively support the historian’s interpretation with clear and logical connections.
  - **Depth of Explanation:**
    - Provides thorough and detailed explanations connecting the evidence to the argument.

- **General Criteria for All Subparts:**
  - **Identify:** Names or states a correct, relevant historical fact.
  - **Describe:** Provides specific details about a historical event or concept beyond isolated terms.
  - **Explain:** Analyzes cause/effect, context, or significance with clear connections to the prompt and historians' arguments.
  - **Compare:** Discusses similarities and/or differences explicitly, if required.
  - **Analyze:** Breaks down the meaning, impact, or significance of the stimulus or broader topic with depth.

### **Output Format for Grading:**

- **Stimulus Reference:**
  - Provide a brief explanation of how the stimulus relates to the question.

- **Feedback for Each Subpart:**

  - **A: (Score: X/1)**  
    *Feedback:* [Detailed feedback...]

  - **B: (Score: X/1)**  
    *Feedback:* [Detailed feedback...]

  - **C: (Score: X/1)**  
    *Feedback:* [Detailed feedback...]

- **Total Score and Summary:**

  - **Total Score:** X/Y (e.g., 2/3 for a 3-part question).
  - **Strengths:** Highlight what the student did well, referencing specific parts of their responses.
  - **Areas for Improvement:** Suggest specific ways to improve their response, focusing on accuracy, relevance, and depth of analysis.
  - **Originality Note:** Comment on the originality and appropriateness of supporting evidence used in Parts B and C, ensuring they reference events not included in the excerpts.

### **Additional Instructions:**

- **Precision Over Presence:** Ensure that points are awarded based on the **quality and accuracy** of the response rather than merely the presence of relevant keywords or events.
- **Alignment with Historians' Arguments:** Pay close attention to whether the student's explanations accurately reflect the specific arguments and interpretations of the historians in the excerpts.
- **Avoid Over-Awarding Points:** Be stringent in evaluating the depth and correctness of the connections made by the student, especially in Parts A and C where nuanced understanding is crucial.
- **Constructive Feedback:** Provide feedback that not only points out deficiencies but also guides the student on how to address them, enhancing their historical analysis skills.
- **Strict Binary Scoring:** Only award a 1 if the response fully meets all criteria; otherwise, award a 0. Do not give partial credit under any circumstances.

### **Examples of Feedback:**

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
    if "summation" in state and state["summation"]:
        return state["summation"]
    else:
        raise ValueError("Summation not found in the final state.")

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
    if "summation" in state and state["summation"]:
        return state["summation"]
    else:
        raise ValueError("Summation not found in the final state.")

