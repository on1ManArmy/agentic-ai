from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

tavily = TavilySearch()


@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.invoke(input=query)


llm = ChatGroq(model="qwen/qwen3.8-27b")
tools = [search]
agent = create_agent(tools=tools, model=llm)


def main():
    print("Hello from ai-agents!")
    result = agent.invoke(
        {"messages": HumanMessage(content="What is weather in Hyderabad, India")}
    )
    structured = result.get("structured_response", None)
    print(structured if structured is not None else result)


if __name__ == "__main__":
    main()
