import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import ToolMessage


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


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b

model_with_tools = model.bind_tools(tools=[multiply])


def call_model(state: MessagesState):
    reply = model_with_tools.invoke(state["messages"])
    return {"messages": [reply]}

tools = [multiply]
tools_by_name = {t.name: t for t in tools}


def call_tools(state: MessagesState):
    last = state["messages"][-1]          # the model's reply, with the tool requests
    results = []
    for call in last.tool_calls:
        tool_fn = tools_by_name[call["name"]]
        output = tool_fn.invoke(call["args"])
        print(f"[tool] {call['name']} {call['args']} -> {output}")
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results}


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if last.tool_calls:
        return "call_tools"               # the model asked for a tool: run it
    return END                            # no tool request: we're done

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("call_tools", call_tools)
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue)
builder.add_edge("call_tools", "call_model")   # the loop
graph = builder.compile()

msg = ""
history = []

while msg != "bye":
    msg = input("> ")

    history.append({"role": "user", "content": msg})

    result = graph.invoke({"messages": history})

    history = result["messages"]
    
    print(result["messages"][-1].content)

