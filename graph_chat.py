import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import ToolMessage, SystemMessage
from langgraph.types import interrupt, Command
from pathlib import Path
import difflib
from langgraph.errors import GraphRecursionError
from langgraph.checkpoint.memory import InMemorySaver
import uuid
import subprocess



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

ROOT = Path.cwd().resolve()
SKIP_DIRS = {".venv", ".git", "node_modules", "__pycache__"}
MAX_READ_CHARS = 20_000
MAX_SEARCH_RESULTS = 50


def safe_path(p: str) -> Path:
    """Turn a model-supplied path into a real path, refusing anything outside ROOT."""
    full = (ROOT / p).resolve()
    if not full.is_relative_to(ROOT):
        raise ValueError(f"'{p}' is outside the project folder.")
    return full


@tool
def list_files(path: str = ".") -> str:
    """List the files and folders inside a folder of the project.

    Args:
        path: Folder to list, relative to the project root. Defaults to the root.
    """
    try:
        folder = safe_path(path)
        if not folder.is_dir():
            return f"Error: '{path}' is not a folder or does not exist."
        entries = []
        for item in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
            if item.name in SKIP_DIRS:
                continue
            entries.append(item.name + ("/" if item.is_dir() else ""))
        return "\n".join(entries) or "(empty folder)"
    except (ValueError, OSError) as e:
        return f"Error: {e}"


@tool
def read_file(path: str) -> str:
    """Read a text file from the project and return it with line numbers.
    Long files are cut off.

    Args:
        path: File to read, relative to the project root.
    """
    try:
        file = safe_path(path)
        if not file.is_file():
            return f"Error: '{path}' is not a file or does not exist."
        with file.open(encoding="utf-8") as f:
            text = f.read(MAX_READ_CHARS + 1)
    except UnicodeDecodeError:
        return f"Error: '{path}' looks like a binary file and can't be read as text."
    except (ValueError, OSError) as e:
        return f"Error: {e}"

    truncated = len(text) > MAX_READ_CHARS
    text = text[:MAX_READ_CHARS]
    numbered = "\n".join(
        f"{i}: {line}" for i, line in enumerate(text.splitlines(), start=1)
    )
    if truncated:
        numbered += f"\n... [cut off after {MAX_READ_CHARS} characters]"
    return numbered or "(empty file)"


@tool
def search_files(query: str, path: str = ".") -> str:
    """Search the project's text files for a string (case-insensitive).
    Returns matching lines as 'file:line: text'.

    Args:
        query: Text to look for.
        path: Folder to search in, relative to the project root. Defaults to the root.
    """
    try:
        start = safe_path(path)
    except ValueError as e:
        return f"Error: {e}"
    if not start.is_dir():
        return f"Error: '{path}' is not a folder or does not exist."

    needle = query.lower()
    matches = []
    for dirpath, dirnames, filenames in start.walk():
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]  # skip noisy folders
        for name in filenames:
            file = dirpath / name
            if not file.resolve().is_relative_to(ROOT):  # symlink pointing outside
                continue
            try:
                with file.open(encoding="utf-8") as f:
                    for line_no, line in enumerate(f, start=1):
                        if needle in line.lower():
                            rel = file.relative_to(ROOT)
                            matches.append(f"{rel}:{line_no}: {line.strip()[:200]}")
                            if len(matches) >= MAX_SEARCH_RESULTS:
                                matches.append(f"... [stopped after {MAX_SEARCH_RESULTS} matches]")
                                return "\n".join(matches)
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable file: skip it
    return "\n".join(matches) or f"No matches for '{query}'."

@tool

def create_file(path:str, content:str) -> str:
    """Create a new file with the given text content. Fails if the file already exists.

    Args: 
        path: path for the new file
        content: the full text to put in the file

    """

    try:
        file = safe_path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("x", encoding="utf-8") as f:
            f.write(content)
        return f"Created {file.relative_to(ROOT)} ({len(content)} characters)."
    except FileExistsError:
        return f"Error: '{path}' already exists. Use a different name or edit the file."
    except (ValueError, OSError) as e:
        return f"Error: {e}"

@tool

def edit_file(path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
    """Edit a file by replacing exact text. old_text must match the file exactly,
    including spaces and indentation, and must be unique unless replace_all is true.
    Read the file first. Do not include line-number prefixes in old_text

    Args:
        path: File to edit, relative to the project root.
        old_text: The exact text to replace.
        new_text: The text to put in its place
        replace_all: Replace every occurrence instead of requiring a single match.
    """

    if not old_text:return "Error: old_text can't be empty. To make a new file, use create_file."
    if old_text == new_text:
        return "Error: old_text and new_text are identical, so nothing would change."

    try:
        file = safe_path(path)
        if not file.is_file():
            return f"Error: '{path}' is not a file or does not exist."
        original = file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: '{path}' looks like a binary file and can't be edited as text."
    except (ValueError, OSError) as e:
        return f"Error: {e}"

    count = original.count(old_text)
    if count == 0:
        return ("Error: old_text was not found. It must match the file exactly, including "
                "indentation. Read the file again and copy the text precisely, without line numbers.")
    if count > 1 and not replace_all:
        return (f"Error: old_text appears {count} times. Include more surrounding lines "
                "to make it unique, or set replace_all to true.")

    updated = original.replace(old_text, new_text)
    try:
        file.write_text(updated, encoding="utf-8")
    except OSError as e:
        return f"Error: {e}"

    rel = file.relative_to(ROOT)
    diff = "".join(difflib.unified_diff(
        original.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=f"a/{rel}", tofile=f"b/{rel}", n=2,
    ))
    return f"Edited {rel} ({count} replacement{'s' if count != 1 else ''}).\n{diff[:3000]}"

@tool

def shell_functions(command: str) -> str:
    """ run a temrinal command in the shell and return the output"""
    
    try:
        result = subprocess.run(command,shell=True,capture_output=True,text=True,timeout=10)

        if result.returncode == 0:
            return result.stdout if result.stdout else "Command completed successfully with no output."
        return f"Error (Exit Code {result.returncode}): {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 10 seconds."
    except Exception as e:
        return f"Exception occurred: {str(e)}"


tools = [list_files, read_file, search_files, create_file, edit_file,shell_functions]
approval_needed = ["create_file", "edit_file","shell_functions"]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)


SYSTEM_PROMPT = (
    f"You are a coding assistant working in the project folder {ROOT}. "
    "Use your tools to look at files before answering questions about the project. "
    "All paths are relative to the project folder. Don't guess what a file contains."
    "Use shell commands when needed and do skip use of shell commands when it's not needed"
    "For editing and creating files use your given tools and use the shell for only shell functions"
    "If you don't have permission to run a command, explain why and then tell the"
    "user how to run it while giving them the appropiate information"
    "Commands have to be non interactive because there's no keyboard"
)


def call_model(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    reply = model_with_tools.invoke(messages)
    return {"messages": [reply]}



def call_tools(state: MessagesState):
    last = state["messages"][-1]
    results = []

    # Ask once if ANY request in this batch needs approval
    if any(call["name"] in approval_needed for call in last.tool_calls):
        summary = "\n".join(
            f"  {call['name']} {str(call['args'])[:300]}" for call in last.tool_calls
        )
        approved = interrupt(f"The agent wants to run:\n{summary}\nAllow?")

        if approved != "yes":
            for call in last.tool_calls:
                results.append(ToolMessage(
                    content="The user rejected this action. Don't retry it; ask what to do instead.",
                    tool_call_id=call["id"],
                ))
            return {"messages": results}

    for call in last.tool_calls:
        tool_fn = tools_by_name[call["name"]]
        output = tool_fn.invoke(call["args"])
        print(f"[tool] {call['name']} {call['args']} -> {str(output)[:200]}")
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
graph = builder.compile(checkpointer=InMemorySaver())




thread_id = str(uuid.uuid4())   # the name of this conversation's save slot

msg = ""
while msg != "bye":
    msg = input("> ")
    if not msg.strip():
        continue

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
    try:
        result = graph.invoke({"messages": [{"role": "user", "content": msg}]}, config)

        while "__interrupt__" in result:
            question = result["__interrupt__"][0].value
            print("\n" + question)
            answer = input("[y/n] ").strip().lower()
            decision = "yes" if answer == "y" else "no"
            result = graph.invoke(Command(resume=decision), config)

    except GraphRecursionError:
        print("Agent stopped: too many steps in one turn. Starting a fresh conversation.")
        thread_id = str(uuid.uuid4())
        continue

    print(result["messages"][-1].content)

