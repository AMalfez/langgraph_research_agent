from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b")