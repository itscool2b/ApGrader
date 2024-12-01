import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import openai
from langchain.agents import Tool, initialize_agent
from langchain.prompts import PromptTemplate
from .pineconesetup import index
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI
load_dotenv()

#pdf reader
#reader = PdfReader("leq.pdf")
#texts = "".join([page.extract_text() for page in reader.pages])
#print(texts)
#embeddings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-large")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#vector database
#store = FAISS.from_texts([texts], embeddings)

#functions
#def get_relevant_documents(query):
   # return store.similarity_search(query, k=5)
def get_relevant_documents(query):
    import openai
    response = client.embeddings.create(query, model="text-embedding-3-small")
    query_embedding = response.data[0].embedding
    results = index.query(query_embedding, top_k=5, include_metadata=True)
    return [result["metadata"]["text"] for result in results["matches"]]

#prompt template
prompt = PromptTemplate.from_template(""" You are an AP US History essay grader following the College Board's rubric. Your task is to evaluate a student's essay with the utmost accuracy, analyzing it against the provided rubric and example graded essays from the AP US History standards. Your evaluation must align precisely with the prompt provided and should reflect the specific requirements and focus outlined in the rubric.

The provided rubric and example essays are included below for your reference:
{relevant_docs}
Student Essay to Grade:
{student_essay}

Grading Instructions:
Analyze the student's essay with precision and based solely on the following criteria from the rubric:
Thesis (0-1 point): Does the essay have a clear, defensible thesis that directly addresses the given prompt and articulates a line of reasoning?
Contextualization (0-1 point): Does the essay situate its argument within a broader historical context relevant to the given prompt?
Evidence (0-2 points):
Is specific, relevant historical evidence used to support the argument?
Is the evidence effectively connected to the thesis and the specific demands of the prompt?
Analysis and Reasoning (0-2 points):
Does the essay demonstrate historical reasoning (causation, comparison, continuity, and change) relevant to the prompt?
Is the argument cohesive, well-reasoned, and consistently tied to the prompt?
Synthesis (0-1 point): Does the essay connect its argument to a different historical period, geographical area, or context, while maintaining relevance to the given prompt?
Use the rubric and examples from the PDF to guide your scoring, ensuring your evaluation is grounded in the specific content and skills the prompt asks for.
Provide clear, detailed feedback based on the rubric, explaining why points were awarded or deducted.
Output Format:
Thesis (0-1 point): [Score and specific feedback]
Contextualization (0-1 point): [Score and specific feedback]
Evidence (0-2 points): [Score and specific feedback]
Analysis and Reasoning (0-2 points): [Score and specific feedback]
Synthesis (0-1 point): [Score and specific feedback]
Total Score (out of 6): [Score]
Feedback Summary:
[Provide a detailed summary of strengths, weaknesses, and specific suggestions for improvement, emphasizing alignment with the given prompt and rubric expectations.] """)

#relevant_shit = get_relevant_documents(query)

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4"
)

tools = [
    Tool(
        name="get rubric and sample essays",
        func = lambda query: get_relevant_documents(query),
        description="Use this tool to retrieve relevant sections of the rubric and example essays for grading."
    )
]


agent = initialize_agent(
    llm=llm,
    hand_parsing_errors=True,
    max_iterations=50,
    tools=tools,
    agent="zero-shot-react-description", 
    verbose=True
)

def evaluate_essay(student_essay):
    
    query = "the entire AP US History LEQ rubric and sample essays"
    relevant_docs = "\n\n".join(get_relevant_documents(query))

    formatted_prompt = prompt.format(
        relevant_docs=relevant_docs,
        student_essay=student_essay
    )

    
    return agent.run(formatted_prompt)