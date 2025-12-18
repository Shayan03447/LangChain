from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel,RunnablePassthrough,RunnableSequence
load_dotenv()
model=ChatOpenAI()

def count(text):
    return len(text.split())

template=PromptTemplate(template='Generate a joke on {topic}',
                        input_variables=['topic'])

parser=StrOutputParser()

joke_gen_chain=RunnableSequence(template, model, parser)
parallel_chain=RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'count':RunnableLambda(count)
    }
)
final_chain=RunnableSequence(joke_gen_chain, parallel_chain)
result=final_chain.invoke({'topic':'Ai'})

print(result)

