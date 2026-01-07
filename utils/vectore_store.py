from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv  
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
loader_multiple_pages = WebBaseLoader(urls)
docs = loader_multiple_pages.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_documents(docs)

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store_explicit_embeddings = AstraDBVectorStore(
    collection_name="test_vector_db_collection",
    embedding=embeddings,
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
)

def retrieve(state):
    """Retrieve documents from the vector store based on the state query."""
    query = state["query"]
    retriever = vector_store_explicit_embeddings.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 1, "score_threshold": 0.5},
    )
    res = retriever.invoke(query)
    return {"query":query, "docs":res}



if __name__ == "__main__":
    print("Adding documents to the vector store...")
    vector_store_explicit_embeddings.add_documents(docs)
    print("Documents added.")
    # print(docs[1])