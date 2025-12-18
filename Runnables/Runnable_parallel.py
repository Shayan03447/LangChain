from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

template1=PromptTemplate(template="Generate a tweet about {topic}",
                        input_variables=['topic'])
template2=PromptTemplate(template="Generate a linkedin post about {topic}",
                        input_variables=['topic'])

model=ChatOpenAI()

parser=StrOutputParser()

parallel_chain=RunnableParallel(
    {
        'tweet':RunnableSequence(template1, model, parser),
        'linkedin':RunnableSequence(template2, model, parser)
    }
)
result=parallel_chain.invoke({'topic':'AI'})
print(result['tweet'])