from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()
model=ChatOpenAI

template=ChatPromptTemplate([
    ('system',"you are the helpfull customer support agent"),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])
chat_history=[]
with open('chat_history.txt') as f:
    chat_history.extend(f.readline())
print(chat_history)

prompt=template.invoke({
    'chat_history':chat_history,'query':"where is my refund"
})
print(prompt)