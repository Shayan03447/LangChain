from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv
load_dotenv()
model=ChatOpenAI()

message=[
    SystemMessage(content="You are a helpfull assistance")
]
while True:
    user_input=input("You: ")
    message.append(HumanMessage(content=user_input))
    if user_input=="Exit":
        break
    result=model.invoke(message)
    message.append(AIMessage(content=result.content))
    print("AI ",result.content)

print(message)