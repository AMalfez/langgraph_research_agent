from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b")

class Router(BaseModel):
    "Decides whether to call wikipedia_search tool or search in vector store."
    datasource: Literal["wikipedia", "vector_store"] = Field(...,
        description="Decide whether to use 'wikipedia' or 'vector_store' based on the user query."
    )

model_with_structure = model.with_structured_output(Router)