from langchain.vectorstores import Redis
from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings
import requests


# Your custom embedding class
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
redis_url = "redis://10.245.33.66:6379"
index_name = "my-index"


def main():
    embedding = OllamaEmbeddings(model='deepseek-r1:1.5b')

    # Monkey patch Redis module check
    Redis._check_redis_module_exist = lambda *args: True

    # Load the existing index from Redis
    vector_store = Redis.from_existing_index(
        embedding=embedding,
        redis_url=redis_url,
        index_name=index_name,
        schema="flat"  # this only works with `langchain.vectorstores.Redis`
    )

    # Build RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=embedding,  # If you have a real LLM (like OpenAI), replace this
        retriever=vector_store.as_retriever(),
        chain_type="stuff"  # or "map_reduce" / "refine"
    )

    query = "What is LangChain?"
    answer = qa.run(query)
    print(f"Q: {query}\nA: {answer}")


if __name__ == "__main__":
    main()
