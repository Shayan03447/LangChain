from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

docs=[
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

def init_vector_store(document, embeddings):
    vector_store=FAISS.from_documents(
        documents=document,
        embedding=embeddings
    )
    return vector_store
def Contextual_retriever(vectorstore, k=3, model="gpt-3.5-turbo"):
    #Base_Retriever
    base_retreiver=vectorstore.as_retriever(search_type="mmr", search_kwargs={
        "k":k,
        "lambda_mult":0.5
    })
    llm=ChatOpenAI(model=model)
    compressor=LLMChainExtractor.from_llm(llm)

    contextualRetriever=ContextualCompressionRetriever(
        base_retriever=base_retreiver,
        base_compressor=compressor
    )
    return contextualRetriever
if __name__=="__main__":
    embeddings=OpenAIEmbeddings()
    store=init_vector_store(document=docs, embeddings=embeddings)
    retriever=Contextual_retriever(vectorstore=store)
    query="What is photosynthesis?"
    result=retriever.invoke(query)
    for i, doc in enumerate(result):
        print(f"\n---Result{i+1}---")
        print(doc.page_content)