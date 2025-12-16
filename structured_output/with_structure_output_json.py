from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing_extensions import Annotated,Optional,Literal
load_dotenv()
model=ChatOpenAI()

json_schema={
    "title":"Ai_Job_Description_Analyzer",
    "type":"object",
    "properties":{
        "job_title":{
            "type":"string",
            "description":"write down the title of the job"
        },
        "required_skills":{
            "type":"array",
            "items":{
                "type":"string"
            },
            "description":"Required mentioned in the discription"
        },
        "Experience_year":{
            "type":"integer",
            "description":"How many year of experience that is required for this job"
        },
        "job_type":{
            "type":"string",
            "enum":["full time","part time","contract"]
        },
        "location_type":{
            "type":"string",
            "enum":["remote","hybrid","onsite"],
            "description":"location type of this role "
        },
        "salary_range":{
            "type":"object",
            "properties":{
                "min_salary":{
                    "type":"integer"
                },
                "max_salary":{
                    "type":"integer"
                }
            },
            "description":"salary range fot this job"
        }

    },
    "required":["job_title","required_skills","job_type"]

}
structure_model=model.with_structured_output(json_schema)
result=structure_model.invoke("""
We are hiring a Senior Python Developer with experience in Django and FastAPI.
The candidate should have at least 4 years of experience and strong knowledge of REST APIs.
This role is remote and offers a salary range of $70,000 to $90,000 per year.
The position is full-time.""")
print(result)