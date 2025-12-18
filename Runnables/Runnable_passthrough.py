from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model=ChatOpenAI()

template1=PromptTemplate(template="write a joke on {topic}",
                        input_variables=["topic"])
template2=PromptTemplate(template="Explain the following joke - {text}",
                        input_varibales=["text"])
parser=StrOutputParser()

joke_gen_chain=RunnableSequence(template1, model, parser)
parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'explaination':RunnableSequence(template2, model, parser)
})
final_chain=RunnableSequence(joke_gen_chain, parallel_chain)
result=final_chain.invoke({'topic':'cricket'})
print(result)