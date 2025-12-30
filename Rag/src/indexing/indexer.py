from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
load_dotenv()
from langchain_community.vectorstores import Chroma

class Indexer:
    def __init__(self, embedding_model=None):
        self.embedding_model=OpenAIEmbeddings()
        self.documents=None
        self.chunks=None
        self.vector_store=None

    # Load Documents
    def load_documents(self, data_path: str):
        loader=PyPDFLoader(data_path)
        self.documents=loader.load()
        print(f"Loaded {len(self.documents)} documents(s)")
        return self.documents
    
    # Splitter
    def chunker(self, chunk_size, chunk_overlap):
        if not self.documents:
            raise ValueError("Load documents first")
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.chunks=splitter.split_documents(self.documents)
        print(f"Split into {len(self.chunks)} chunks")
        return self.chunks
    
    # Vector_store
    def create_vectorstore(self):
        if not self.chunks:
            raise ValueError("Split document first")
        self.vector_store=Chroma.from_documents(
            self.chunks,
            self.embedding_model
        )
        print("Vector store created")
        return self.vector_store
        