from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.schema import Document

def wikipedia_search(state):
    """Searches the wikipedia for the given query and returns a summary of the results.

    Args:
        query: The search query string.
    """
    query = state["query"]
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    docs = Document(page_content=wikipedia.run(query))

    return {docs: docs, query: query}