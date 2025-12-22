from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv  import load_dotenv
from langchain_core.runnables import RunnableSequence
load_dotenv()

model=ChatOpenAI()

template=PromptTemplate(template="Write a summary for the following poem - \n {poem}",
                        input_variables=['poem'])

parser=StrOutputParser()

loader=TextLoader("Document_Loader/Cricket.txt",encoding='utf=8')
docs=loader.load()

chain=template | model | parser
result=chain.invoke({
    'poem':docs[0].page_content
})
print(result)
