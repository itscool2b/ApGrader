import os
from dotenv import load_dotenv
import openai
from langchain.agents import Tool, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Import Pinecone setup function to get the index
from .pineconesetup import get_index
index = get_index()

# Define OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

# Function to fetch relevant documents from Pinecone
def get_relevant_documents(query):
    # Generate embedding for the query using OpenAI's API
    response = openai.Embedding.create(input=[query], model="text-embedding-ada-002")
    query_embedding = response['data'][0]['embedding']
    
    # Query Pinecone index for relevant matches
    results = index.query(query_embedding, top_k=5, include_metadata=True)
    
    # Extract and return text from the metadata of the results
    return [match["metadata"]["text"] for match in results["matches"]]

# Prompt template for evaluating essays
prompt = PromptTemplate.from_template("""
You are an AP US History essay grader following the College Board's rubric. 
Your task is to evaluate a student's essay with the utmost accuracy, analyzing 
it against the provided rubric and example graded essays from the AP US History standards. 
Your evaluation must align precisely with the prompt provided and should reflect 
the specific requirements and focus outlined in the rubric.

The provided rubric and example essays are included below for your reference:
{relevant_docs}

Student Essay to Grade:
{student_essay}

Grading Instructions:
Analyze the student's essay with precision and based solely on the following criteria from the rubric:
- **Thesis (0-1 point):** Does the essay have a clear, defensible thesis that directly addresses the given prompt and articulates a line of reasoning?
- **Contextualization (0-1 point):** Does the essay situate its argument within a broader historical context relevant to the given prompt?
- **Evidence (0-2 points):** 
    - Is specific, relevant historical evidence used to support the argument?
    - Is the evidence effectively connected to the thesis and the specific demands of the prompt?
- **Analysis and Reasoning (0-2 points):** 
    - Does the essay demonstrate historical reasoning (causation, comparison, continuity, and change) relevant to the prompt?
    - Is the argument cohesive, well-reasoned, and consistently tied to the prompt?
- **Synthesis (0-1 point):** Does the essay connect its argument to a different historical period, geographical area, or context, while maintaining relevance to the given prompt?

Use the rubric and examples from the PDF to guide your scoring, ensuring your evaluation is grounded in the specific content and skills the prompt asks for.
Provide clear, detailed feedback based on the rubric, explaining why points were awarded or deducted.

Output Format:
- **Thesis (0-1 point):** [Score and specific feedback]
- **Contextualization (0-1 point):** [Score and specific feedback]
- **Evidence (0-2 points):** [Score and specific feedback]
- **Analysis and Reasoning (0-2 points):** [Score and specific feedback]
- **Synthesis (0-1 point):** [Score and specific feedback]
- **Total Score (out of 6):** [Score]
- **Feedback Summary:** [Provide a detailed summary of strengths, weaknesses, and specific suggestions for improvement, emphasizing alignment with the given prompt and rubric expectations.]
""")

# Define the LLM for LangChain with OpenAI's GPT-4
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4"
)

# Define tools for the agent
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

# Function to evaluate an essay
def evaluate_essay(student_essay):
    query = "the entire AP US History LEQ rubric and sample essays"
    relevant_docs = "\n\n".join(get_relevant_documents(query))
    
    # Format the prompt for evaluation
    formatted_prompt = prompt.format(
        relevant_docs=relevant_docs,
        student_essay=student_essay
    )
    
    # Use OpenAI's ChatCompletion API for grading
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AP US History essay grader."},
            {"role": "user", "content": formatted_prompt},
        ],
    )
    
    # Return the content of the response
    return response['choices'][0]['message']['content']
