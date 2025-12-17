from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
load_dotenv()
model=ChatOpenAI()

template=PromptTemplate(template="generate a report on {topic}",
                        input_variables=['topic'])
template1=PromptTemplate(template="generate a  5 pointer summary from the following text \n {text}",
                        input_variables=['text'])

parser=StrOutputParser()

chain=template | model | parser | template1 | model |parser
result=chain.invoke({"topic":"knee pain"})
print(result)