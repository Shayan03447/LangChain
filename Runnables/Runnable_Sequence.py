from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model=ChatOpenAI()

template1=PromptTemplate(template="write a joke on {topic}",
                        input_variables=["topic"])
template2=PromptTemplate(template="Explain the following joke - {text}",
                        input_varibales=["text"])
parser=StrOutputParser()

#chain=template1 | model | parser | template2 | model | parser
#result=chain.invoke({'topic':"machine_learning"})
#print(result)
chain=RunnableSequence(template1,model,parser,template2,model,parser)
result=chain.invoke({'topic':'machine_learning'})
print(result)