from langchain.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

@tool
def wikipedia_search(query: str) -> str:
    """Searches the wikipedia for the given query and returns a summary of the results.

    Args:
        query: The search query string.
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    wikipedia.run(query)
    return wikipedia.run(query)