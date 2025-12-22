from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
model=ChatOpenAI()

# Document loader 
loader=PyPDFLoader("E:\Internships\LangChain\Document_Loader\dl-curriculum.pdf")
docs=loader.load()

#Splitter
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks=splitter.split_documents(docs)


# Prompt
template=PromptTemplate(template="""
Answer the following question - {question} from the following - {text}""",
input_variables=["question","text"])

#Parser
parser=StrOutputParser()

question="key component mentioned in dl-curriculum"

# Chain
chain=template | model | parser
result=chain.invoke({
    'question':question,
    'text':chunks
})
print(result)