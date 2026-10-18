from langchain.tools import tool
from openrouter import OpenRouter
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("HACK_CLUB_API_KEY")
url = os.getenv("ROUTER_URL")
model = os.getenv("MODEL")

for name, value in [("HACK_CLUB_API_KEY", api_key), ("ROUTER_URL", url), ("MODEL", model)]:
    if not value:
        print(name,"is missing")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b

client = OpenRouter(
    api_key=api_key,
    server_url=url,
)

history = []

while True:
    user_text = input("\nYou: ").strip()
    if user_text.lower() in ("quit", "exit"):
        break
    if not user_text:
        continue

    history.append({"role": "user", "content": user_text})

    try:

        stream = client.chat.send(
            model=model,
            messages=history,
            stream=True,
            tools=[multiply]
        )

        print("Bot: ", end="", flush=True)
        reply = ""
        for event in stream:
            piece = event.choices[0].delta.content if event.choices else None
            if piece:  
                print(piece, end="", flush=True)
                reply += piece
        print()


        history.append({"role": "assistant", "content": reply})

    except Exception as e:
        print(f"\nRequest failed: {type(e).__name__}")
        print("Status:", getattr(e, "status_code", "unknown"))
        print("Details:", getattr(e, "body", None) or e)
        history.pop()  # 