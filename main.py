from importlib.metadata import version
import os

from dotenv import load_dotenv
from pydantic import SecretStr
load_dotenv()

from langchain_core import __version__ as core_version
graph_version = version('langgraph')

from langchain_google_genai import ChatGoogleGenerativeAI


print(f"LangChain Core Version: {core_version}")
print(f"LangGraph Version: {graph_version}")


def main():
    llm = ChatGoogleGenerativeAI(
        model = "gemini-3.1-flash-lite",
        api_key = SecretStr(os.environ["GOOGLE_API_KEY"]),

    )
    response = llm.invoke("Write a poem about the beauty of nature.")

    print(response.text)

if __name__ == "__main__":
    main()
    