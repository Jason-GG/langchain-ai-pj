from langchain_community.vectorstores import Redis
from langchain.chains import RetrievalQA
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


def main():
    embedding = OllamaEmbeddings(model='deepseek-r1:1.5b')

    # No schema passed here
    vector_store = Redis.from_existing_index(
        embedding=embedding,
        redis_url="redis://10.245.33.66:6379",
        index_name="my-index"
    )

    qa = RetrievalQA.from_chain_type(
        llm=embedding,  # Replace this with real LLM if needed
        retriever=vector_store.as_retriever(),
        chain_type="stuff"
    )

    query = "What is LangChain?"
    result = qa.run(query)
    print(result)


if __name__ == "__main__":
    main()
