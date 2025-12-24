from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
load_dotenv()

doc1=Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"}
)
doc2=Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Mumbai Indians"}
)
doc3=Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"}
)
doc4=Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"}
)
doc5=Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"}
)
docs=[doc1,doc2,doc3,doc4,doc5]
# initailize the vector store

def init_vector_store(docs, embeddings, persist_dir="my_chromaDB", collection_name="sample"):
    store=Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name
        )
    store.persist()
    return store

def search(store, query, k):
    return store.similarity_search(query, k=k)
def search_with_score(store, query, k):
    return store.similarity_search_with_score(query, k=k)

def metadata_filter(store, query, metadata_filter, k):
    return store.similarity_search_with_score(
        query=query,
        k=k,
        filter=metadata_filter
    )
def update_vector_document(store, document_id, new_doc: Document):
    return store.update_document(document_id=document_id, document=new_doc)



# Main Execution
if __name__=="__main__":
    # Initialize embedding
    embedding=OpenAIEmbeddings()
    # create vector store
    store=init_vector_store(docs=docs,embeddings=embedding) 
    print(store.get(include=['embeddings','documents','metadatas']))
    # Search 
    #result=search_with_score(store=store, query=" how is all #rounder", k=2)
    # metadata filter
    #result=metadata_filter(store=store, query='',metadata_filter={"team": "Chennai Super Kings"}, k=1)
    updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
    )
    update_vector_document(store=store, document_id='09a39dc6-3ba6-4ea7-927e-fdda591da5e4',new_doc=updated_doc1)
    #print(result)

    
    