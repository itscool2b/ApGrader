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
            top_k=10,
            include_metadata=True
        )
        return [match["metadata"]["text"] for match in results["matches"]]

    except Exception as e:
        raise RuntimeError(f"Error in embedding or querying Pinecone: {e}")


prompt = PromptTemplate.from_template("""You are an AP US History essay grader using the College Board's updated LEQ rubric from 2023. Your task is to evaluate a student's Long Essay Question (LEQ) strictly based on the rubric provided. All feedback, scores, and analysis must directly reference the rubric and examples provided below.

Do not introduce any criteria not explicitly included in the rubric.

The scoring system is out of 6 points, and your grading must align with realistic standards used by AP US History graders. Use knowledge from the retrieved documents to ensure historical accuracy and alignment with College Board-approved standards. Be fair but rigorous in your grading, awarding points where criteria are reasonably met. However, the provided examples are to be used as references, not strict benchmarks, for evaluation.

The essay should only receive credit when it meets the exact criteria outlined in the rubric. If the essay does not fulfill the requirements for a point, do not award partial credit or attempt to rationalize its inclusion.

Additional Emphasis:
The retrieved documents include essays labeled with scores 1, 2, 3, 4, 5, and 6. These sample essays provide valuable references for understanding scoring standards and may be used as helpful guides to inform your evaluation. Use them as supplemental guidance, allowing flexibility in your evaluation while considering them as informative examples.

Do not take spelling and grammar into account when evaluating the essay. Focus solely on the content and how it aligns with the rubric criteria. As long as the meaning of the essay is clear, spelling and grammar errors should not impact the evaluation.

Evaluation Process:

The rubric, textbook, and related examples retrieved are provided below:
{relevant_docs}

Student Essay to Grade:
{student_essay}

Evaluation Criteria:

Contextualization (0-1 point):
Provide feedback on how well the essay establishes historical context as per the rubric. Explain whether the essay earned the point by explicitly detailing its strengths or, if not, describe the shortcomings and reference specific rubric requirements that were unmet.
Evaluate the essay based on its ability to provide historical context, as outlined in the rubric.

Thesis / Claim (0-1 point):
Provide detailed feedback on the clarity and relevance of the thesis or claim. If the point was awarded, explain why the thesis met the rubric's standards. If not, identify the issues and suggest specific improvements.
Assess the thesis or claim for clarity, historical relevance, and alignment with rubric criteria.

Evidence (0-2 points):

Specific Evidence: Provide detailed feedback on the use of specific historical evidence. If the point was awarded, highlight the strengths and alignment with rubric criteria. If not, explain what was lacking and reference the rubric requirements for improvement.
Evidence Supporting Argument: Offer detailed feedback on whether the evidence effectively supports the essay’s arguments. If the point was awarded, specify how the evidence met rubric criteria. If not, clarify the gaps and suggest ways to improve.

Analysis and Reasoning (0-2 points):

Historical Reasoning: Provide detailed feedback on the essay's use of historical reasoning. Highlight strengths if the point was awarded, or explain how the essay fell short if not, directly referencing the rubric for guidance.
Complex Understanding: Offer detailed feedback on the essay’s demonstration of nuanced and complex understanding. If the point was awarded, specify the strengths. If not, identify the gaps and connect the feedback to the rubric for actionable improvement.

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
Provide a realistic and strict summary of the essay’s strengths, weaknesses, and areas for improvement. For each scoring category, detail where points were gained or lost, explicitly referencing the rubric to explain how the essay met or failed to meet the criteria. Offer clear, actionable feedback to guide improvement in future essays. Feedback must directly reference the rubric criteria and provide actionable suggestions for improvement. Focus on alignment with the precise historical accuracy and analytical depth expected in AP US History essays, using the retrieved documents as supplemental guidance.

Strict Grading Policy:
Always emphasize that the total score is out of 6 points, and apply no leniency in evaluating the essay. Award points only when the essay fully satisfies the rubric’s requirements. Marginal or implied fulfillment of a criterion is not sufficient for credit. Align your grading with the rigorous standards used by actual AP US History graders.

Minimum Score Rule:
The worst score you can give is a total of 1. If the essay is completely off-topic, incoherent, or incomplete, it should still receive at least 1 point. The only exception is if the student submits absolutely nothing—in that case, the score is 0. Otherwise, the minimum score is always 1.


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
        description="Retrieve the rubric (`leq_rubric`), textbook (`amsco_textbook`), and example essays (labeled by score) from the vector database. Use these for grading."
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

        relevant_docs = agent.invoke("Retrieve AP US History rubric, examples, and textbook for grading.")
        
        
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

