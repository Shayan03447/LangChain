from  langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()
embedding=OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

document=["Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."]


query="Tell me about the virat kohli"

#craete embeeding
doc_embedding=embedding.embed_documents(document)
query_embedding=embedding.embed_query(query)

score=cosine_similarity([query_embedding], doc_embedding)[0]

index, score=sorted(list(enumerate(score)),key=lambda x:x[1])[-1]
print(query)
print(document[index])
print("Similarity score is :", score)

