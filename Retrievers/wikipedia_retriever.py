from langchain_community.retrievers import WikipediaRetriever

retriver=WikipediaRetriever(top_k_results=2, lang='en')
query="what is the geopolitical history of india and pakistan from the perspective of a chinese"
docs=retriver.invoke(query)
print(docs)
    