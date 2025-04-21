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

import requests
from langchain.embeddings.base import Embeddings

class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        return response.json()["embedding"]



def init_vector_store():
    embedding_model = OllamaEmbeddings()
    # Connect to Redis
    # vector_store = Redis(
    #     redis_url="redis://mce-vecotr-db.dx0ahi.ng.0001.use1.cache.amazonaws.com:6379",
    #     index_name="ollama-index",
    #     embedding=embedding_model
    # )
    vector_store = Redis.from_existing_index(
        redis_host="mce-vecotr-db.dx0ahi.ng.0001.use1.cache.amazonaws.com",
        redis_port=6379,
        embedding=embedding_model)

    return vector_store

vector_store = init_vector_store()

def load_and_chunk_contents_of_blog():
    # Load and chunk contents of the blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)
    # Define prompt for question-answering
    prompt = hub.pull("rlm/rag-prompt")
    print(prompt)


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def compile_application_and_test():
    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph


if __name__ == "__main__":
    graph = compile_application_and_test()
    state = graph.invoke({"question": "What is the main topic of this blog post?"})
    print(state["answer"])