from openrouter import OpenRouter
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("HACK_CLUB_API_KEY")
url = os.getenv("ROUTER_URL")
model = os.getenv("MODEL")

client = OpenRouter(
    api_key=api_key,
    server_url=url,
)

response = client.chat.send(
    model=model,
    messages=[
        {"role": "user", "content": "hi"}
    ],
)

print(response.choices[0].message.content)