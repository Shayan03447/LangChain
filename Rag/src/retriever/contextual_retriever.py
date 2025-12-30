from langchain_openai import ChatOpenAI
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv
load_dotenv()

class ContextualRetriever:
    def __init__(self, vector_store=None, top_k=5):
        self.vector_store=vector_store
        self.top_k=top_k
        self.retriever=None

    def setup_retriever(self, model="gpt-3.5-turbo"):
        if not self.vector_store:
            raise ValueError("Vector store not set")
        base_retriever=self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k":self.top_k,
                "lambda_mult":0.5
            })
        llm=ChatOpenAI(model=model)
        compressor=LLMChainExtractor.from_llm(llm=llm)
        self.retriever=ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=compressor
        )
        print(f"Contextual Retriever setup with top {self.top_k} results")
        return self.retriever

    def query(self, query_text, context_text):
        if not self.retriever:
            raise ValueError("Retriever not found")
        final_query=query_text
        if context_text:
            final_query=f"Context:{context_text} \nQuery: {query_text}"

        results=self.retriever.invoke(final_query)
        return results



        

