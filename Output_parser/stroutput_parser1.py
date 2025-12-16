# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# load_dotenv()
# model=ChatOpenAI()

# template1=PromptTemplate(template="Write down the detail report on {topic}",
#                          input_variables=["topic"])
# template2=PromptTemplate(template="write a 5 line summary on the following text. /n{text}",
#                          input_variables=["text"])
# parser=StrOutputParser()

# chain=template1 | model | parser | template2 | model | parser
# result=chain.invoke({"topic":'black hole '})
# print(result)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model=ChatOpenAI()

template=PromptTemplate(template="""
You are an AI job description analyzer.

Given the job description below, extract the information and return ONLY
one single line in this exact format:

<Job Title> | <Experience in years> | <Job Type>

Rules:
- Do NOT add explanations
- Do NOT add new lines
- Do NOT add extra text

Job Description:
{job_description}
""",
input_variables=['job_description'])

parser=StrOutputParser()
chain=template | model |parser
result=chain.invoke({
    "job_description":"""
    We are hiring a Backend Engineer.
    Required skills: Python, FastAPI, PostgreSQL.
    Experience required: 3 years.
    Job type: Full-time"""
})
print(result)

