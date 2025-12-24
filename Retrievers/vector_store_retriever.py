from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv()

docs = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

def init_vector_store(docs,embeddings, persist_dir="my_vectorstore", collection_name="my_collection"):
    store=Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name
    )
    store.persist()
    return store

def create_retriever(store, k=2):
    retriever=store.as_retriever(search_kwargs={"k":k})
    return retriever

if __name__=="__main__":
    embedding=OpenAIEmbeddings()
    store=init_vector_store(docs=docs, embeddings=embedding, )
    retriever=create_retriever(store=store, k=2)
    query="Who is viral_koli"
    docs=retriever._get_relevant_documents(query)
    for i, doc in enumerate(docs, 1):
        print(f"Document {i}: {doc.page_content}")

	
    