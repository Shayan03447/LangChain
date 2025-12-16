from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
model=ChatOpenAI()

template=PromptTemplate(template="Write a detailed report on {topic}",
                        input_variables=["topic"])

template1=PromptTemplate(template="write a 5 line summary on the following text. /n {text}",
                         input_variables=["text"])

prompt1=template.invoke({'topic':'black hole'})
result=model.invoke(prompt1)
prompt2=template1.invoke({'text':result.content})
result1=model.invoke(prompt2)
print(result1.content)