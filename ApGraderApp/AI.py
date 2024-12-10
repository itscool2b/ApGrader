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
All feedback, scores, and analysis must directly reference the rubric and examples retrieved from the vector database.
Do not introduce any criteria not explicitly included in the rubric.

The scoring system is out of 6 points, and your grading must align with realistic standards used by AP US History graders. Use only knowledge from actual AP US History textbooks, such as AMSCO, or other College Board-approved materials. Ensure your evaluation is grounded in historical accuracy and reflects the expectations set by the AP US History course.

The prompt and essay will be given to you at the same time in the same text block. Pay close attention to both, and ensure your evaluation is thorough and accurate. Avoid any ambiguity and focus solely on the criteria specified in the rubric.

The rubric and related examples retrieved from the vector database are provided below:
{relevant_docs}

Student Essay to Grade:
{student_essay}

Evaluation Criteria:
Contextualization (0-1 point):
Refer to the rubric to evaluate this category.

Thesis / Claim (0-1 point):
Refer to the rubric to evaluate this category.

Evidence (0-2 points):

Specific Evidence: Refer to the rubric to evaluate this subcategory.
Evidence Supporting Argument: Refer to the rubric to evaluate this subcategory.
Analysis and Reasoning (0-2 points):

Historical Reasoning: Refer to the rubric to evaluate this subcategory.
Complex Understanding: Refer to the rubric to evaluate this subcategory.
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
Provide a realistic summary of the essay’s strengths, weaknesses, and areas for improvement. Feedback must directly reference the rubric criteria and provide actionable suggestions for improvement. Focus on alignment with the historical accuracy and analytical depth expected in AP US History essays, drawing on approved materials such as AMSCO or other College Board-endorsed resources.

Strictly use the rubric retrieved from the vector database for all evaluations. Always emphasize that the total score is out of 6 points and align your grading with actual AP US History standards.
""")


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

        relevant_docs = agent.run("Retrieve AP US History rubric and examples for grading.")
        
        
        formatted_prompt = prompt.format(
            relevant_docs=relevant_docs,
            student_essay=student_essay
        )
        
        
        response = agent.run(formatted_prompt)
        return response

    except Exception as e:
        raise RuntimeError(f"Error in evaluating essay: {e}")
    

def test_pinecone_query():
    query = "What is the AP US History rubric for contextualization?"
    documents = get_relevant_documents(query)
    print("Retrieved Documents:", documents)

