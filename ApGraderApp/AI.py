import os
from dotenv import load_dotenv
import openai
from langchain.agents import Tool, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI
from .pineconesetup import get_index

# Load environment variables
load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)
# Initialize Pinecone index
index = get_index()


def get_relevant_documents(query):
    """Retrieve relevant documents from Pinecone based on the query embedding."""
    try:
        # Generate embedding for the query using the older API format
        
        response = client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
        
        )
        query_embedding = response.data[0].embedding

        # Query Pinecone index for relevant documents
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        return [match["metadata"]["text"] for match in results["matches"]]

    except Exception as e:
        raise RuntimeError(f"Error in embedding or querying Pinecone: {e}")

# Define the prompt template
prompt = PromptTemplate.from_template("""
You are an AP US History essay grader following the College Board's LEQ rubric. 
Your task is to evaluate a student's Long Essay Question (LEQ) based strictly on the rubric stored in the vector database. 
Use only the information from the provided rubric and related examples retrieved from the vector store for grading. 
All feedback, scoring, and analysis must be grounded explicitly in the rubric criteria.

The retrieved rubric and sample essays are included below for reference:
{relevant_docs}

Student Essay to Grade:
{student_essay}

Grading Instructions:
Evaluate the student's LEQ solely based on the following criteria from the rubric. Provide scores and detailed feedback for each category exactly as structured in the rubric:

1. **Thesis (0-1 point):**
    - Does the essay have a clear and defensible thesis that directly addresses all parts of the prompt?
    - Does the thesis establish a line of reasoning?

2. **Contextualization (0-1 point):**
    - Does the essay situate its argument within a broader historical context relevant to the prompt?

3. **Evidence (0-2 points):**
    - **Specific Evidence (1 point):** Does the essay provide specific and relevant historical evidence?
    - **Evidence Supporting an Argument (1 point):** Is the evidence used effectively to support the thesis or argument?

4. **Analysis and Reasoning (0-2 points):**
    - **Historical Reasoning (1 point):** Does the essay demonstrate the use of historical reasoning skills (causation, comparison, continuity, and change) relevant to the prompt?
    - **Complex Understanding (1 point):** Does the essay demonstrate a complex understanding of the historical development addressed in the prompt?

5. **Synthesis (0-1 point):**
    - Does the essay extend the argument by explaining connections to a different historical period, development, or context?

Output Format:
- **Thesis (0-1 point):** [Score and specific feedback]
- **Contextualization (0-1 point):** [Score and specific feedback]
- **Evidence (0-2 points):**
    - Specific Evidence: [Score and specific feedback]
    - Evidence Supporting an Argument: [Score and specific feedback]
- **Analysis and Reasoning (0-2 points):**
    - Historical Reasoning: [Score and specific feedback]
    - Complex Understanding: [Score and specific feedback]
- **Synthesis (0-1 point):** [Score and specific feedback]
- **Total Score (out of 6):** [Score]

**Feedback Summary:**
Provide a detailed summary of the essay's strengths, weaknesses, and suggestions for improvement. Ensure your feedback explicitly references the rubric criteria and clearly identifies how the essay does or does not meet the rubric's expectations.

Strictly adhere to the rubric criteria retrieved from the vector store for all evaluations.
""")

# Initialize the LLM
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4"
)

# Define tools for LangChain agent
tools = [
    Tool(
        name="get rubric and sample essays",
        func=lambda query: "\n\n".join(get_relevant_documents(query)),
        description="Retrieve relevant sections of the rubric and example essays for grading."
    )
]

# Initialize the agent
agent = initialize_agent(
    llm=llm,
    tools=tools,
    agent="zero-shot-react-description",
    verbose=True
)

def evaluate_essay(student_essay):
    """Evaluate the student's essay using the OpenAI GPT-4 model and the rubric."""
    try:
        # Query for relevant documents
        query = "the entire AP US History LEQ rubric and sample essays"
        relevant_docs = "\n\n".join(get_relevant_documents(query))

        # Format the prompt with the rubric and essay
        formatted_prompt = prompt.format(
            relevant_docs=relevant_docs,
            student_essay=student_essay
        )

        # Call the OpenAI ChatCompletion API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AP US History essay grader."},
                {"role": "user", "content": formatted_prompt},
            ],
        )

        # Extract and return the response content
        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"Error in evaluating essay: {e}")
