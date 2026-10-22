import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, MessagesState, START, END


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


model = ChatOpenRouter(
    model=model,
    base_url=url,
    api_key=api_key
)


def call_model(state: MessagesState):
    reply = model.invoke(state["messages"])
    return {"messages": [reply]}

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)
graph = builder.compile()

msg = ""
history = []

while msg != "bye":
    msg = input("> ")

    history.append({"role": "user", "content": msg})

    result = graph.invoke({"messages": history})

    history = result["messages"]
    
    print(result["messages"][0].content)

