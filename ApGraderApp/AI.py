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
You are an AP US History essay grader using the College Board's updated LEQ rubric from 2023. 
Your task is to evaluate a student's Long Essay Question (LEQ) strictly based on the rubric provided. 
All feedback, scores, and analysis must directly reference the rubric and examples retrieved from the vector database. 
Do not introduce any criteria not explicitly included in the rubric.

The rubric and related examples retrieved from the vector database are provided below:
{relevant_docs}

Student Essay to Grade:
{student_essay}

### Evaluation Criteria:
1. **Contextualization (0-1 point):**
   Refer to the rubric to evaluate this category.

2. **Thesis / Claim (0-1 point):**
   Refer to the rubric to evaluate this category.

3. **Evidence (0-2 points):**
   - Specific Evidence: Refer to the rubric to evaluate this subcategory.
   - Evidence Supporting Argument: Refer to the rubric to evaluate this subcategory.

4. **Analysis and Reasoning (0-2 points):**
   - Historical Reasoning: Refer to the rubric to evaluate this subcategory.
   - Complex Understanding: Refer to the rubric to evaluate this subcategory.

### Output Format:
- **Contextualization (0-1 point):** [Score with feedback]
- **Thesis / Claim (0-1 point):** [Score with feedback]
- **Evidence (0-2 points):**
    - Specific Evidence: [Score with feedback]
    - Evidence Supporting Argument: [Score with feedback]
- **Analysis and Reasoning (0-2 points):**
    - Historical Reasoning: [Score with feedback]
    - Complex Understanding: [Score with feedback]
- **Total Score (out of 6):** [Score]

### Feedback Summary:
Provide a summary of the essay’s strengths, weaknesses, and areas for improvement. Feedback must directly reference the rubric criteria.

Strictly use the rubric retrieved from the vector database for all evaluations.
""")

# Initialize the LLM
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o-mini"
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
        response = agent.run({
            "query": "You are an AP US History essay grader.",
            "student_essay": student_essay
        })
        return response
        # Query for relevant documents
        ##relevant_docs = "\n\n".join(get_relevant_documents(query))

        # Format the prompt with the rubric and essay
        #formatted_prompt = prompt.format(
            #relevant_docs=relevant_docs,
            #student_essay=student_essay
        #)

        # Call the OpenAI ChatCompletion API
        #response = client.chat.completions.create(
            #model="gpt-4o",
            #messages=[
                #{"role": "system", "content": "You are an AP US History essay grader."},
                #{"role": "user", "content": formatted_prompt},
            #],
        #)

        # Extract and return the response content
        #return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"Error in evaluating essay: {e}")
