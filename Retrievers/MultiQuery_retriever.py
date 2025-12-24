from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

model=ChatOpenAI()

docs=[
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]
def init_vector_store(document, embeddings):
    vector_store=FAISS.from_documents(
        documents=document,
        embedding=embeddings
    )
    return vector_store

def Multiquery_retriever(vectorstore, k=3, model="gpt-3.5-turbo"):
    base_retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":k, "lambda_mult":0.5})
    llm=ChatOpenAI(model=model)
    multiquery_retriever=MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    return multiquery_retriever

if __name__=="__main__":
    embeddings=OpenAIEmbeddings()
    vector_store=init_vector_store(document=docs, embeddings=embeddings)
    retriever=Multiquery_retriever(vectorstore=vector_store)
    query="How to improve energy levels and maintain balance?"
    result=retriever.invoke(query)
    for i, doc in enumerate(result):
        print(f"\n---Result{i+1}---")
        print(doc.page_content)
        print("*"*150)



