# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# load_dotenv()
# model=ChatOpenAI()
# parser=JsonOutputParser()

# template= PromptTemplate(template="Give me the 5 fact about {topic} \n {format_instruction}",
#                          input_variables=[],
#                          partial_variables={'format_instruction':parser.get_format_instructions()})
# chain=template | model | parser
# result=chain.invoke({'topic':'black hole '})
# print(result)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()
model=ChatOpenAI()

parser=JsonOutputParser()

template=PromptTemplate(template="""
You are an AI job description analyzer.

Extract the following fields from the job description and return ONLY
a valid JSON object with these keys:

- job_title (string)
- experience_year (number)
- job_type (string)
- location_type (string)

Rules:
- Return ONLY JSON
- No explanation
- No extra text

Job Description:
{job_description} \n
{format_instruction} 
""",
input_variables=["job_description"],
partial_variables={'format_instruction':parser.get_format_instructions()})

chain = template | model | parser
result= chain.invoke({
    "job_description":"""
    We are hiring a Backend Engineer.
    Required skills: Python, FastAPI, PostgreSQL.
    Experience required: 3 years.
    Job type: Full-time.
    Location: Remote."""
})
print(result)