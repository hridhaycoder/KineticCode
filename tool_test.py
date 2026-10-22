import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter


load_dotenv()

api_key = os.getenv("HACK_CLUB_API_KEY")
url = os.getenv("ROUTER_URL")
model = os.getenv("MODEL")


missing_vars = False
for name, value in [("HACK_CLUB_API_KEY", api_key), ("ROUTER_URL", url), ("MODEL", model)]:
    if not value:
        print(f"ERROR: {name} is missing in your .env file!")
        missing_vars = True

if missing_vars:
    exit(1)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b



llm = ChatOpenRouter(
    model=model,
    base_url=url,
    api_key=api_key
)


agent = create_agent(
    model=llm,
    tools=[multiply]
)


response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Can you use the tool to check if 5*5 is six"}
    ]
})


final_message = response["messages"][-1].content
print("\nAgent Response:")
print(final_message)
