import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from week1.scraper import fetch_website_contents
from week1.day1.api_key_validator import validate_api_key

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

validate_api_key(api_key)

# client = OpenAI()
openai = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # A placeholder key is required but ignored by Ollama
)

system_prompt = """
You summarize movie websites. Find the 10 latest movies,
prioritize release date and reputable sources,
and return for each: title, release date, genre,
short spoiler-free summary, IMDb/rating if available,
and a source link. Keep it concise and accurate.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""

user_prompt_prefix = """
Find and summarize the **top 10 latest movies** available on the provided movie website.

Prioritize the newest releases and return:

1. Movie title
2. Release date
3. Genre
4. Rating
5. A short, spoiler-free summary
6. Source URL

Keep each summary concise and rank the movies from newest to oldest.

"""


def messages_for(website: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_prefix + website},
    ]


def summarize(url: str) -> str:
    website = fetch_website_contents(url)
    response = openai.chat.completions.create(
        model="gemma3:270m ",
        messages=messages_for(website),
        stream=True,
    )
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print()

# website_url = input("Enter the movie website URL: ")
# print(summarize(website_url))

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://edwarddonner.com"
    print(summarize(url))