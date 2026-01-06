from llm import model_with_structure
from prompts import ROUTER_PROMPT
def router(state):
    """A simple router function that decides the datasource based on the query."""
    query = state["query"]
    messages = [
        (
            "system",
            ROUTER_PROMPT,
        ),
        ("human", query),
    ]
    res = model_with_structure.invoke(messages)
    
    if res.datasource == "wikipedia":
        return "wikipedia_search"
    else:
        return "vector_db_search"


if __name__ == "__main__":
    router("Tell me about ai agent.")