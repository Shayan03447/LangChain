from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

def init_vector_store(document, embeddings):
    vector_store=FAISS.from_documents(
        documents=document,
        embedding=embeddings
    )
    return vector_store
def create_retriever(vectorstore):
    retriever=vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":3,
            "lambda_mult":0.5
        }
    )
    return retriever

if __name__=="__main__":
    embedding=OpenAIEmbeddings()
    vectorstore=init_vector_store(document=docs, embeddings=embedding)
    retriever=create_retriever(vectorstore=vectorstore)
    query="what is langchain"
    result=retriever.invoke(query)
    for i, doc in enumerate(result):
        print(f"\n---Result {i+1}---")
        print(doc.page_content)

