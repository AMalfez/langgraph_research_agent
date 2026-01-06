from llm import model
from utils.wikipedia_search import wikipedia_search
from router import router

# Augment the LLM with tools
tools = [wikipedia_search]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

from typing_extensions import TypedDict
class MessagesState(TypedDict):
    query: str
    docs: list[str]

from langgraph.graph import StateGraph, START, END



# Build workflow
agent_builder = StateGraph(MessagesState)

agent_builder.add_node("wikipedia_search", wikipedia_search)
agent_builder.add_conditional_edges(START, router, {
    "wikipedia_search": "wikipedia_search",
    "vector_db_search": "vector_db_search"
})

# Compile the agent
agent = agent_builder.compile()


# from IPython.display import Image, display
# # Show the agent
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

if __name__ == "__main__":
    # Invoke
    from langchain.messages import HumanMessage
    messages = [HumanMessage(content="World war 2")]
    messages = agent.invoke({"messages": messages})
    print(messages["messages"][-1].content)