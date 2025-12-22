from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
model=ChatOpenAI()

template=PromptTemplate(template="""Answer the following question - \n {question} from the following text - \n {text}""",
                        input_variables=['question','text'])
parser=StrOutputParser()
url='https://en.wikipedia.org/wiki/Machine_learning'
loader=WebBaseLoader(url)
docs=loader.load()

chain=template | model | parser
question="History of machine_learning"

result=chain.invoke({'question':question, 'text':docs[0].page_content})
print(result)