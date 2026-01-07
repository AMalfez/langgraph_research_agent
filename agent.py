# from llm import model
from utils.wikipedia_search import wikipedia_search
from router import router
from utils.vectore_store import retrieve as vector_db_search

from typing_extensions import TypedDict
class MessagesState(TypedDict):
    query: str
    docs: list[str]

from langgraph.graph import StateGraph, START, END



# Build workflow
agent_builder = StateGraph(MessagesState)

agent_builder.add_node("wikipedia_search", wikipedia_search)
agent_builder.add_node("vector_db_search", vector_db_search)
agent_builder.add_conditional_edges(START, router, {
    "wikipedia_search": "wikipedia_search",
    "vector_db_search": "vector_db_search"
})
agent_builder.add_edge("wikipedia_search", END)
agent_builder.add_edge("vector_db_search", END)

# Compile the agent
agent = agent_builder.compile()


# from IPython.display import Image, display
# # Show the agent
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

if __name__ == "__main__":
    # Invoke
    content="World war 2"
    messages = agent.invoke({"query": content})
    print(messages["docs"][-1].content)