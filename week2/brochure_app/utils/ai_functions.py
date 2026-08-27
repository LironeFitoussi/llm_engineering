from openai import OpenAI
import json
import os
from dotenv import load_dotenv
from utils import validate_api_key, fetch_website_contents
from utils.prompts import link_system_prompt, get_links_user_prompt, brochure_system_prompt, get_brochure_user_prompt

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
validate_api_key(api_key, "OPENAI_API_KEY")
# Initialize the OpenAI API client
openai = OpenAI()
# Set the models to use
MODEL = "gpt-5-nano"
BROCHURE_MODEL = "gpt-4.1-mini"

def select_relevant_links(system_prompt, user_prompt, url):
    print(f"Selecting relevant links for {url} by calling {MODEL}")

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    links = json.loads(result)

    print(f"Found {len(links['links'])} relevant links")

    return links


def fetch_page_and_relevant_links(url):
    """Fetch the landing page plus the content of every AI-selected relevant link."""
    contents = fetch_website_contents(url)
    relevant_links = select_relevant_links(link_system_prompt, get_links_user_prompt(url), url)

    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    for link in relevant_links["links"]:
        result += f"\n\n### Link: {link['type']}\n"
        result += fetch_website_contents(link["url"])
    return result


def stream_brochure(company_name, url):
    """Generate a brochure for the company and stream it to the terminal as it arrives."""
    page_contents = fetch_page_and_relevant_links(url)
    user_prompt = get_brochure_user_prompt(company_name, url, page_contents)

    print(f"\nGenerating brochure for {company_name} by calling {BROCHURE_MODEL}...\n")

    stream = openai.chat.completions.create(
        model=BROCHURE_MODEL,
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )

    response = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            response += content
            print(content, end="", flush=True)
    print()
    return response