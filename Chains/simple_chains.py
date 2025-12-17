from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
model=ChatOpenAI()
parser=StrOutputParser()

template=PromptTemplate(template="Generate the 5 lines summary for the following {topic}",
                        input_variables=["topic"])
chain = template | model | parser
result=chain.invoke({
    "topic":"lstm"
})
print(result)

chain.get_graph().print_ascii()