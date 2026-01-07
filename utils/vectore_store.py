from dotenv import load_dotenv  
load_dotenv()
from vector_store_setup import vector_store_explicit_embeddings

def retrieve(state):
    """Retrieve documents from the vector store based on the state query."""
    query = state["query"]
    retriever = vector_store_explicit_embeddings.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 1, "score_threshold": 0.5},
    )
    res = retriever.invoke(query)
    return {"query":query, "docs":res}

