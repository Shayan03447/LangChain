from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
load_dotenv()
model=ChatOpenAI()

schema=[
    ResponseSchema(name='fact 1',description='fact 1 about the topic'),
    ResponseSchema(name='fact 2',description='fact 2 about the topic'),
    ResponseSchema(name='fact 3',description='fact 3 about the topic'),
]
parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(template='Give three fact about {topic} \n {format_instruction}',
                        input_variables=["topic"],
                        partial_variables={"format_instruction":parser.get_format_instruction()})
chain=template | model |parser
result=chain.invoke({'topic':'black hole'})
print(result)
