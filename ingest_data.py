from langchain.vectorstores import Redis
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

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
    embedding = OpenAIEmbeddings()

    # ---- Ingest documents into Redis ----
    vector_store = Redis.from_documents(
        documents=docs,
        embedding=embedding,
        redis_url=redis_url,
        index_name=index_name
    )

    print(f"Successfully ingested {len(docs)} documents into Redis index '{index_name}'")
