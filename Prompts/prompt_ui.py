from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
import streamlit as st
load_dotenv()
#model
model=ChatOpenAI()

st.header("Cover Letter Generator")

Position=st.selectbox("Select Job Title/Position ",["Data Scientist","Ai_Engineer","Machine_Learning_Engineer"])
Style=st.selectbox("Select Tone",["Professional","Cascual","Creative"])
length=st.selectbox("Select Length",["Short","Medium","Detailed"])


template=load_prompt('template.json')

if st.button("Cover Letter"):
    chain=template | model
    result=chain.invoke({
        "job_title":Position,
        "tone":Style,
        "length":length
    })
    st.write(result.content)



