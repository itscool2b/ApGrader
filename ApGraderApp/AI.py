import os
from dotenv import load_dotenv
import openai
from langchain.agents import Tool, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

from openai import OpenAI
from .pineconesetup import get_index

load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

index = get_index()


def get_relevant_documents(query):
    """Retrieve relevant documents from Pinecone based on the query embedding."""
    try:
       
        
        response = client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
        
        )
        query_embedding = response.data[0].embedding

        
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        return [match["metadata"]["text"] for match in results["matches"]]

    except Exception as e:
        raise RuntimeError(f"Error in embedding or querying Pinecone: {e}")


prompt = PromptTemplate.from_template(""" You are an AP US History essay grader using the College Board's updated LEQ rubric from 2023.
Your task is to evaluate a student's Long Essay Question (LEQ) strictly based on the rubric provided.
All feedback, scores, and analysis must directly reference the rubric retrieved from the vector database.
Do not introduce any criteria not explicitly included in the rubric.

The scoring system is out of 6 points, and your grading must align with realistic standards used by AP US History graders. Apply slightly higher standards than average AP graders, ensuring that all points awarded demonstrate strong adherence to the rubric. Use knowledge from actual AP US History textbooks, such as AMSCO, or other College Board-approved materials. Be strict and apply no leniency in your grading. The essay should only receive credit when it meets the criteria outlined in the rubric. If the essay does not fulfill the requirements for a point, do not award partial credit or attempt to rationalize its inclusion.

Do not take spelling and grammar into account when evaluating the essay. Focus solely on the content and how it aligns with the rubric criteria. As long as the meaning of the essay is clear, spelling and grammar errors should not impact the evaluation.

The prompt of the LEQ will be included in the same text block as the student's essay and will be clearly highlighted for your reference in the provided PDF. Ensure you pay attention to this prompt as it directly dictates the context and focus of your evaluation. Do not overemphasize the prompt, but make sure it is clearly understood and integrated into your grading process.

The prompt and essay will be given to you at the same time in the same text block. Pay close attention to both, and ensure your evaluation is thorough and accurate. Avoid any ambiguity and focus solely on the criteria specified in the rubric.

The rubric is retrieved from the vector database are provided below:
{relevant_docs}

Student Essay to Grade:
{student_essay}

Evaluation Criteria:

Contextualization (0–1 point):

1 point: Awarded only if the essay provides a detailed and accurate description of the broader historical context directly relevant to the prompt. The context should be explicitly connected to the argument, demonstrating how it informs the topic at hand.
0 points: Do not award the point for vague, general, or unrelated contextual information. Mere passing references without clear connections to the argument are insufficient.

Thesis / Claim (0–1 point):

1 point: Awarded only if the essay presents a specific, historically defensible thesis that directly addresses all aspects of the prompt and establishes a clear line of reasoning. The thesis must be located in the introduction or conclusion.
0 points: Do not award the point for restatements of the prompt, overgeneralized statements, or claims lacking a clear line of reasoning.
Evidence (0–2 points):

Specific Evidence (1 point): Awarded only if the essay provides at least two specific, relevant pieces of historical evidence that directly support the argument. Generalizations or broad statements without specific details do not merit this point.
Evidence Supporting Argument (1 point): Awarded only if the essay effectively uses the specific evidence to support a cohesive argument in response to the prompt. The connections between the evidence and the argument must be explicit and well-explained.
Analysis and Reasoning (0–2 points):

Historical Reasoning (1 point): Awarded only if the essay employs a historical reasoning skill (e.g., comparison, causation, continuity and change) to structure an argument that directly addresses the prompt. The reasoning should be coherent and consistently applied throughout the essay.
Complex Understanding (1 point): Awarded only if the essay demonstrates a nuanced understanding of the historical development in question. This can be achieved by analyzing multiple variables, considering diverse perspectives, or making connections across different historical contexts. Superficial or simplistic analysis does not qualify for this point.



Output Format:
Contextualization (0-1 point): [Score with feedback]
Thesis / Claim (0-1 point): [Score with feedback]
Evidence (0-2 points):
Specific Evidence: [Score with feedback]
Evidence Supporting Argument: [Score with feedback]
Analysis and Reasoning (0-2 points):
Historical Reasoning: [Score with feedback]
Complex Understanding: [Score with feedback]
Total Score (out of 6): [Score]
Feedback Summary:
Provide a realistic and strict summary of the essay’s strengths, weaknesses, and areas for improvement. Feedback must directly reference the rubric criteria and provide actionable suggestions for improvement. Focus on alignment with the precise historical accuracy and analytical depth expected in AP US History essays, drawing on approved materials such as AMSCO or other College Board-endorsed resources.

Strict Grading Policy: Always emphasize that the total score is out of 6 points, and apply no leniency in evaluating the essay. Award points only when the essay satisfies the rubric’s requirements. Marginal or implied fulfillment of a criterion is not sufficient for credit. Align your grading with rigorous standards used by actual AP US History graders.
Also note this. Ensure the total score is calculated as the sum of the points the student receives in the following categories: thesis, evidence, contextualization, and complex understanding and analysis.
"""












)


llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o"
)


tools = [
    Tool(
        name="get rubric and sample essays",
        func=lambda query: "\n\n".join(get_relevant_documents(query)),
        description="Retrieve relevant sections of the rubric and example essays for grading. Use the entire thing for grading for each and every part."
    )
]


agent = initialize_agent(
    llm=llm,
    tools=tools,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)

def evaluate_essay(student_essay):
    """Evaluate the student's essay using the OpenAI GPT-4 model and the rubric."""
    try:

        relevant_docs = agent.invoke("Retrieve AP US History rubric and examples for grading.")
        
        
        formatted_prompt = prompt.format(
            relevant_docs=relevant_docs,
            student_essay=student_essay
        )
        
        
        response = agent.invoke(formatted_prompt)
        return response

    except Exception as e:
        raise RuntimeError(f"Error in evaluating essay: {e}")
    

def test_pinecone_query():
    query = "What is the AP US History rubric for contextualization?"
    documents = get_relevant_documents(query)
    print("Retrieved Documents:", documents)

