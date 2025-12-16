from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from typing_extensions import Optional
load_dotenv()
model=ChatOpenAI()

class SalaryRange(BaseModel):
    min_salary: Optional[int]
    max_salary: Optional[int]

class job(BaseModel):
    job_title: str
    experience_year: int
    job_type: str
    location_type: str
    salary_range: SalaryRange

parser=PydanticOutputParser(pydantic_object=job)

template=PromptTemplate(template="""
You are an AI job description analyzer.

Extract the following fields from the job description and return ONLY a valid JSON object:

- job_title (string)
- experience_year (integer)
- job_type (string)
- location_type (string)
- salary_range (object)
    - min_salary (integer)
    - max_salary (integer)

Rules:
- Return ONLY JSON
- No extra text or explanation
- Follow the format instructions below exactly

Job Description:
{job_description}
\n
{format_instructions}
""",
input_variables=["job_description"],
partial_variables={'format_instructions':parser.get_format_instructions()})

chain= template | model | parser
result=chain.invoke({
    "job_description":"""
We are hiring a Backend Engineer.
Required skills: Python, FastAPI, PostgreSQL.
Experience required: 3 years.
Job type: Full-time.
Location: Remote"""
})
print(result)
