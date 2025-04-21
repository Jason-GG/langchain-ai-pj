from langchain.vectorstores import redis
from langchain.schema import Document
# ollama_embeddings.py
from langchain_core.embeddings import Embeddings
import requests



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



# ---- Configuration ----
redis_url = "redis://mce-vecotr-db.dx0ahi.ng.0001.use1.cache.amazonaws.com:6379"
index_name = "my-index"

# Example documents (can come from files, DB, etc.)
docs = [
    Document(page_content="LangChain is a framework for building LLM applications."),
    Document(page_content="Redis is an in-memory data structure store, used as a database, cache, and message broker."),
    Document(page_content="Ollama enables local inference of LLMs like LLaMA and Mistral.")
]


if __name__ == "__main__":
    # ---- Embedding model ----
    # If you’re not using OpenAI, you can replace this with HuggingFaceEmbeddings or others
    embedding = OllamaEmbeddings(model='deepseek-r1:1.5b')

    # ---- Ingest documents into Redis ----
    # command MODULE LIST is not supported by Redis Labs - monkey patch it
    redis._check_redis_module_exist = lambda *args: True
    vector_store = redis.from_documents(
        documents=docs,
        embedding=embedding,
        redis_url=redis_url,
        index_name=index_name
    )

    print(f"Successfully ingested {len(docs)} documents into Redis index '{index_name}'")
