from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()
# Model
model=ChatOpenAI()

# Prompt
template=PromptTemplate(template="""
Answer the question - \n {question} from the following text - \n {text}""",
input_variables=["question","text"])

# Parser
parser=StrOutputParser()

# Webbase loader
url='https://en.wikipedia.org/wiki/Machine_learning'
loader=WebBaseLoader(url)
docs=loader.load()

# splitter

splitter=CharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=40,
    separator=''
)

chunk=splitter.split_documents(docs)

chain=template | model | parser
question='history of machine learning'
result=chain.invoke({
    'question':question,
    'text':chunk[0].page_content
}) 
print(result)