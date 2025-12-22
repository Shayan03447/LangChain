from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("Document_Loader/dl-curriculum.pdf")
docs=loader.load()
print(docs[2].metadata)