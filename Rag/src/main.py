from indexing.indexer import Indexer
from retriever.contextual_retriever import ContextualRetriever

#Initialize indexr
indexer=Indexer()
docs=indexer.load_documents(data_path=r"E:\Internships\LangChain\Rag\data\Sample-Handbook.pdf")
chunks=indexer.chunker(chunk_size=500, chunk_overlap=50)
vector_store=indexer.create_vectorstore()

# Retriever
Retriever=ContextualRetriever(vector_store=vector_store)
Retriever.setup_retriever()
#Query
Query_text="list of offences of misconduct "
Context_text="user asking for offences"
result=Retriever.query(query_text=Query_text, context_text=Context_text)
Structured_results=[]
for doc in result:
    Structured_results.append({
        "content":doc.page_content,
        "metadata":doc.metadata
    })
print(Structured_results)



