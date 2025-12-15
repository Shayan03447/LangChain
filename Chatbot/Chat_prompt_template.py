from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
load_dotenv()
model=ChatOpenAI

template=ChatPromptTemplate([
    #SystemMessage(content="You are a helpfull {domain} expert"),
    #HumanMessage(content="Explain in simple terms, what is {topic}")
    # Created a tuple in the list
    ('system','You are a helpfull {domain} expert'),
    ('human','Explain in simple terms, what is {topic}')

])
prompt=template.invoke({
    'domain':'cricket','topic':'legside'
})
print(prompt)