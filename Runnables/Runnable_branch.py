from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model=ChatOpenAI()

template_email=PromptTemplate(template="""
You are a support classifier.
Classify the email into one category:
- refund
- complaint
- general

Email:
{email}

Return only one word.""",
input_variables=['email'])



template_complain=PromptTemplate(template="""
You are a customer support agent.
Apologize and acknowledge the complaint:

{email}""",
input_variables=['email'])

template_refund=PromptTemplate(template="""
You are a refund support agent.
Write a polite refund response for this email:

{email}""",
input_variables=['email'])

template_general=PromptTemplate(template="""
You are a helpful support assistant.
Reply to this general inquiry:

{email}""",
input_variables=['email'])



parser=StrOutputParser()

classifier_chain=RunnableSequence(template_email, model, parser)


Runnable_branch=RunnableBranch(
   (lambda x: x == "refund", template_refund | model | parser),
    (lambda x: x == "complaint", template_complain | model | parser),
    template_general | model | parser
)

final_chain=RunnableSequence(classifier_chain,Runnable_branch)
email="""
Hi, I ordered a phone last week.
It arrived damaged and I want a refund."""

result=final_chain.invoke({'email':email})
#result=final_chain.invoke({'email':email})
print(result)


