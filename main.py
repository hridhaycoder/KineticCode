import os
from dotenv import load_dotenv  # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore
from langchain_core.messages import HumanMessage  # type: ignore


load_dotenv()


llm = ChatOpenAI(
    model="anthropic/claude-sonnet-5.5",                     
    openai_api_key=os.getenv("HACK_CLUB_API_KEY"),  
    openai_api_base="https://ai.hackclub.com/proxy/v1" 
)

while True:
    content = input("Enter your prompt: ")

    messages = [
        HumanMessage(content=content)
    ]

    response = llm.invoke(messages)
    print(response.content)
