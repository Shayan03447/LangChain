from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from typing_extensions import Literal
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableBranch, RunnableLambda


load_dotenv()
model=ChatOpenAI()

parser=StrOutputParser()

class FeedBack(BaseModel):
    sentiment:Literal["positive","negative"]=Field(description="Give the sentiment of the feedback")

parser2=PydanticOutputParser(pydantic_object=FeedBack)

template1=PromptTemplate(template="""classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}""",
                        input_variables=["feedback"],
                        partial_variables={'format_instruction':parser2.get_format_instructions()})
classifier_chain=template1 | model | parser2

template2=PromptTemplate(template="""
Write an appropriate response to this positive feedback \n {feedback}""",
input_variables=["feedback"])

template3=PromptTemplate(template="""
Write an appropriate response to this negative feedback \n {feedback}""")

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive',template2|model|parser),
    (lambda x:x.sentiment=="negative", template3| model|parser),
    RunnableLambda(lambda x: "could not find sentiment")

)
chain=classifier_chain | branch_chain
result=chain.invoke({'feedback':"this is a beautifull phone"})
print(result)