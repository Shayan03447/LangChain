from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()


llm=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.7)

template=PromptTemplate(template="Suggest a catchy blog title about {topic}",
input_variables=["topic"])

topic=input("Enter a topic:   ")

# prompt=template.invoke({"topic":topic})
# blog_title=llm.invoke(prompt)
# print(blog_title)
# chain=template | topic
# result=chain.invoke({'topic':topic})
# print(result)
chain=template | llm
topic=input('Enter the topic: ')
result=chain.invoke({'topic':topic})
print(result.content
      )