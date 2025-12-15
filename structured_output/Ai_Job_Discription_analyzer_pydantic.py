from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Annotated, Optional, Literal

load_dotenv()
model=ChatOpenAI()


class SalaryRange(BaseModel):
    min_salary: int
    max_salary: int

class JobAnalysis(BaseModel):
    job_title: str= Field(description="write down the title of the job")
    required_skills: list[str]=Field(description="required skill that are present in this discription")
    experience_years: int= Field(discrition="how many years of experience that is required for this job")
    job_type: Literal["full time","part time","contract"]
    location_type:Literal["onsite","remote","hybrid"]
    salary_range:SalaryRange

structured_model=model.with_structured_output(JobAnalysis)

result=structured_model.invoke("""
We are hiring a Senior Python Developer with experience in Django and FastAPI.
The candidate should have at least 4 years of experience and strong knowledge of REST APIs.
This role is remote and offers a salary range of $70,000 to $90,000 per year.
The position is full-time.""")
structured_dict=dict(result)
print(structured_dict)