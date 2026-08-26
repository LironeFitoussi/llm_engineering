# practical things
import os
from dotenv import load_dotenv
from validator.validation import validate_api_key
from openai import OpenAI
# load the environment variables
load_dotenv(override=True)


# ------------------------------------------------------------
# openai via openai
# ------------------------------------------------------------

# # get the API key
# api_key = os.getenv('OPENAI_API_KEY')

# # validate the API key
# validate_api_key(api_key, "OPENAI_API_KEY")

# # create the OpenAI client
# openai = OpenAI(api_key=api_key)

# # create the model
# model = openai.chat.completions.create(
#     model="gpt-5-nano", 
#     messages=[
#         {
#             "role": "user",
#             "content": "Tell me a fun fact"
#         }
#     ]
# )

# print(model.choices[0].message.content)


# ------------------------------------------------------------
# gemini via openai
# ------------------------------------------------------------

# gemini_api_key = os.getenv('GEMINI_API_KEY')
# gemini = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=gemini_api_key)

# validate_api_key(gemini_api_key, "GEMINI_API_KEY")

# model = gemini.chat.completions.create(
#     model="gemini-3.1-flash-lite",
#     messages=[
#         {
#             "role": "user",
#             "content": "Tell me a fun fact"
#         }
#     ]
# )

# print(model.choices[0].message.content)

# ------------------------------------------------------------
# ollama via openai
# ------------------------------------------------------------

# pre-test ollama
# import requests
# content = ""
# response = requests.get("http://localhost:11434").content
# print(response)


# ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


# model = ollama.chat.completions.create(
#     # model="llama3.2",
#     model="deepseek-r1:1.5b",
#     messages=[
#         {
#             "role": "user",
#             "content": "Tell me a fun fact"
#         }
#     ]
# )

# print(model.choices[0].message.content)

# ------------------------------------------------------------
# Implementing web scraper
# ------------------------------------------------------------

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scraper import fetch_website_contents

ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "llama3.2"

system_prompt = """
You are a smart webpage summarizer.
Ignore navigation, ads, cookie banners, and other boilerplate.
Focus on the main content: what the page is about, key facts, and any news or announcements.
Keep the summary short and accurate. Respond in markdown.
long answer should be less than 1000 characters and include some data from the feched data.
Do not wrap the markdown in a code block.
"""

user_prompt_prefix = """
Here are the contents of a website.
Provide a short, clear summary of this page.

"""


def messages_for(website: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_prefix + website},
    ]


def summarize(url: str) -> None:
    print(f"Fetching {url}...\n")
    website = fetch_website_contents(url)
    response = ollama.chat.completions.create(
        model=MODEL,
        messages=messages_for(website),
        stream=True,
    )
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print()


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else input("Enter a webpage URL: ").strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    summarize(url)
