from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated,Optional
load_dotenv()
model=ChatOpenAI()

class NewsAnalysis(TypedDict):
    company_name:Annotated[str,"Extract the company name"]
    product_type:Annotated[str,"What is the type of product"]
    release_date:Annotated[str,"write the release date "]
    key_capability:Annotated[str, "What was the key capability parest in the text"]
    effected_industries:Annotated[list[str], "List of industries affected by this product"]
    event_name:Annotated[str,"what is the event name"]

struture_response=model.with_structured_output(NewsAnalysis)
result=struture_response.invoke("""
OpenAI announced a new AI model focused on reasoning tasks.
The model was released on March 2025 and is expected to significantly improve complex problem solving.
Experts believe it will impact industries such as healthcare, finance, and education.
The announcement was made during a global AI conference""")
print(result)