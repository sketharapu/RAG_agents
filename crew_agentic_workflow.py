import os
from crewai_tools import PDFSearchTool
from crewai_tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai import Crew, Task, Agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import requests

load_dotenv()


llm= ChatOpenAI(
    openai_api_base = "https://api.groq.com/openai/v1",
    openai_api_key = os.getenv("GROQ_API_KEY"),
    model_name= "llama3-8b-8192",
    temperature=0.1,
    max_tokens = 1000,
)

pdf_url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
response = requests.get(pdf_url)

with open('attenstion_is_all_you_need.pdf', 'wb') as file:
    file.write(response.content)


# Create a RAG tool variable to pass PDF

rag_tool = PDFSearchTool(pdf = 'attenstion_is_all_you_need.pdf',
                         config = dict(
                             llm = dict(
                                 provider = 'groq',
                                 config = dict(
                                     model = "llama3-8b-8192",
                                     temperature = 0.5,
                                 ),
                             ),
                         embedder = dict(
                             provider = "huggingface", 
                             config = dict(
                                 model="BAAI/bge-small-en-v1.5",
                             ),
                         ),
                    )
                )
#result = rag_tool.run("How did self-attention mechanism evolve in large language model?")
#print("Search result:", result)

web_search_tool = TavilySearchResults(k = 3)
#web_search_result = web_search_tool.run("What is self-attention mechanism in large language models?")
#print("Search result:", web_search_result)

# Define a Tool

@tool
def router_tool(question):
    """Router Function"""
    if 'self-attention' in question:
        return 'vectorstore'
    else:
        return 'web_search'
    
#Create Agents to work with

Router_Agent = Agent(
    role='Router',
    goal='Route user question to a vectorstore or web search',
    backstory=("You are an expert at routing a user question to a vectorstore or web search."
    "Use the vectorstore for questions on concept related to Retrieval-Augmented Generation."
    "You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search."
  ),
    verbose=True,
    allow_delegation = True,
    llm=llm,
)

Retriever_Agent = Agent(
    role="Retriever",
    goal="Use the information retrieved from the vectorstore to answer the question",
    backstory=(
    "You are an assistant for question-answering tasks."
    "Use the information present in the retrieved context to answer the question."
    "You have to provide a clear concise answer."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

Grader_Agent = Agent(
    role='Answer Grader',
    goal='Filter out erroneous retrievals',
    backstory=(
    "You are a grader assessing relevance of a retrieved document to a user question."
    "If the document contains keywords related to the user question, grade it as relevant."
    "It does not need to be a stringent test.You have to make sure that the answer is relevant to the question."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

hallucination_grader = Agent(
    role="Hallucination Grader",
    goal="Filter out hallucination",
    backstory=(
        "You are a hallucination grader assessing whether an answer is grounded in / supported by a set of facts."
        "Make sure you meticulously review the answer and check if the response provided is in alignmnet with the question asked"
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

answer_grader = Agent(
    role="Answer Grader",
    goal="Filter out hallucination from the answer.",
    backstory=(
        "You are a grader assessing whether an answer is useful to resolve a question."
        "Make sure you meticulously review the answer and check if it makes sense for the question asked"
        "If the answer is relevant generate a clear and concise response."
        "If the answer gnerated is not relevant then perform a websearch using 'web_search_tool'"
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

## Defining Tasks
router_task = Task(
    description=("Analyse the keywords in the question {question}"
    "Based on the keywords decide whether it is eligible for a vectorstore search or a web search."
    "Return a single word 'vectorstore' if it is eligible for vectorstore search."
    "Return a single word 'websearch' if it is eligible for web search."
    "Do not provide any other premable or explaination."
    ),
    expected_output = ("Give a binary choise 'websearch' or 'vectorstore' based on the question"
                       "Do not provide any other premable or explanation."),
    agent=Router_Agent,
    tools=[router_tool],
)

retriever_task = Task(
    description=("Based on the response from the router task extract information for the question {question} with the help of the respective tool."
    "Use the web_serach_tool to retrieve information from the web in case the router task output is 'websearch'."
    "Use the rag_tool to retrieve information from the vectorstore in case the router task output is 'vectorstore'."
    ),
    expected_output=("You should analyse the output of the 'router_task'"
    "If the response is 'websearch' then use the web_search_tool to retrieve information from the web."
    "If the response is 'vectorstore' then use the rag_tool to retrieve information from the vectorstore."
    "Return a claer and consise text as response."),
    agent=Retriever_Agent,
    context=[router_task],
   #tools=[retriever_tool],
)

grader_task = Task(
    description=("Based on the response from the retriever task for the quetion {question} evaluate whether the retrieved content is relevant to the question."
    ),
    expected_output=("Binary score 'yes' or 'no' score to indicate whether the document is relevant to the question"
    "You must answer 'yes' if the response from the 'retriever_task' is in alignment with the question asked."
    "You must answer 'no' if the response from the 'retriever_task' is not in alignment with the question asked."
    "Do not provide any preamble or explanations except for 'yes' or 'no'."),
    agent=Grader_Agent,
    context=[retriever_task],
)

hallucination_task = Task(
    description=("Based on the response from the grader task for the quetion {question} evaluate whether the answer is grounded in / supported by a set of facts."),
    expected_output=("Binary score 'yes' or 'no' score to indicate whether the answer is sync with the question asked"
    "Respond 'yes' if the answer is in useful and contains fact about the question asked."
    "Respond 'no' if the answer is not useful and does not contains fact about the question asked."
    "Do not provide any preamble or explanations except for 'yes' or 'no'."),
    agent=hallucination_grader,
    context=[grader_task],
)

answer_task = Task(
    description=("Based on the response from the hallucination task for the quetion {question} evaluate whether the answer is useful to resolve the question."
    "If the answer is 'yes' return a clear and concise answer."
    "If the answer is 'no' then perform a 'websearch' and return the response"),
    expected_output=("Return a clear and concise response if the response from 'hallucination_task' is 'yes'."
    "Perform a web search using 'web_search_tool' and return ta clear and concise response only if the response from 'hallucination_task' is 'no'."
    "Otherwise respond as 'Sorry! unable to find a valid response'."),
    context=[hallucination_task],
    agent=answer_grader,
    #tools=[answer_grader_tool],
)

# Define a flow for the use case

rag_crew= Crew(
    agents=[Router_Agent, Retriever_Agent, Grader_Agent, hallucination_grader, answer_grader],
    tasks = [router_task, retriever_task, grader_task, hallucination_task, answer_task],
    verbose=True
)

# Ask any query
inputs ={"question":"How does self-attention mechanism help large language models?"}

# Kick off the agent pipeline

result=rag_crew.kickoff(inputs=inputs)
