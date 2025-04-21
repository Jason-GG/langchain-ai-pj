import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import Redis
# vector_store = InMemoryVectorStore(embeddings)
from langchain.chains import RetrievalQA
import requests
from langchain.embeddings.base import Embeddings

redis_url = "redis://10.245.33.66:6379"

class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "nomic-embed-text", endpoint: str = "http://localhost:11434"):
        self.model = model
        self.endpoint = endpoint

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        response = requests.post(
            f"{self.endpoint}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]


def get_qa_chain():
    embedding = OllamaEmbeddings(model='deepseek-r1:1.5b')
    vector_store = Redis.from_existing_index(
        embedding=embedding,
        redis_url=redis_url,
        index_name="my-index",
        index_schema="flat_schema.json" # or "hnsw" depending on what you used when creating the index
    )

    # Create a retrieval-based QA chain
    qa_chain_v = RetrievalQA.from_chain_type(
        llm=embedding,
        chain_type="map_reduce",
        retriever=vector_store.as_retriever()
    )
    return qa_chain_v


if __name__ == "__main__":
    qa_chain = get_qa_chain()
    query = "What is the information about Document 1?"

    # Get the answer using RAG
    result = qa_chain.run(query)
    print(result)