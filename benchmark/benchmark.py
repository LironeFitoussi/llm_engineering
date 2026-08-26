"""
Benchmark: naive vs. optimized `fetch_website_links`

Compares two versions of the link-extraction step used before asking an
LLM to pick "relevant" links for a company brochure:

  naive     - the instructor's version: keeps every href as-is
              (duplicates + '#anchor' links included)
  optimized - the student's version: dedups hrefs with a set() and drops
              pure '#anchor' links before they ever reach the prompt

For each site in sites.py we:
  1. Fetch the page once and parse the anchors once (fair comparison -
     only the *filtering logic* differs, not the network call).
  2. Build the exact system/user prompt used in utils/prompts.py for
     both link lists.
  3. Count prompt tokens locally with tiktoken (free, no network).
  4. Optionally call the real OpenAI API (gpt-5-nano) for both variants
     and read back the actual usage.prompt_tokens / completion_tokens
     and wall-clock latency.

Results are written to benchmark/results.csv and benchmark/results.json,
and a summary table is printed to stdout.

Usage:
    python benchmark.py            # live API calls (default)
    python benchmark.py --no-live  # tiktoken counting only, no API calls
    python benchmark.py --limit 5  # only the first 5 sites
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import requests
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

from benchmark.sites import SITES

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

MODEL = "gpt-5-nano"

LINK_SYSTEM_PROMPT = """
    You are provided with a list of links found on a webpage.
    You are able to decide which of the links would be most relevant to include in a brochure about the company,
    such as links to an About page, or a Company page, or Careers/Jobs pages.
    You should respond in JSON as in this example:

    {
        "links": [
            {"type": "about page", "url": "https://full.url/goes/here/about"},
            {"type": "careers page", "url": "https://another.full.url/careers"}
        ]
    }
"""


def build_user_prompt(url, links):
    prompt = f"""
Here is the list of links on the website {url} -
Please decide which of these are relevant web links for a brochure about the company,
respond with the full https URL in JSON format.
Do not include Terms of Service, Privacy, email links.

Links (some might be relative links):

"""
    prompt += "\n".join(links)
    return prompt


def naive_links(soup):
    """Instructor's version: no dedup, no '#' filtering, only drops falsy hrefs."""
    links = [link.get("href") for link in soup.find_all("a")]
    return [link for link in links if link]


def optimized_links(soup):
    """Student's version: dedup via set(), drop pure '#anchor' links."""
    links = set([link.get("href") for link in soup.find_all("a") if link.get("href")])
    return [link for link in links if link and not link.startswith("#")]


def get_encoding():
    try:
        return tiktoken.encoding_for_model(MODEL)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_tokens(encoding, system_prompt, user_prompt):
    # Rough but consistent proxy for chat prompt tokens: encode system + user text.
    return len(encoding.encode(system_prompt)) + len(encoding.encode(user_prompt))


def call_openai(client, system_prompt, user_prompt):
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    elapsed = time.perf_counter() - start
    usage = response.usage
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "latency_s": round(elapsed, 3),
    }


def run(limit, live):
    encoding = get_encoding()
    client = None
    if live:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY not found in .env - cannot run live benchmark.")
        client = OpenAI(api_key=api_key)

    sites = SITES[:limit] if limit else SITES
    rows = []

    for i, url in enumerate(sites, 1):
        print(f"[{i}/{len(sites)}] {url}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(resp.content, "html.parser")
        except Exception as exc:
            print(f"  skipped (fetch error: {exc})")
            rows.append({"url": url, "status": f"fetch_error: {exc}"})
            continue

        naive = naive_links(soup)
        optimized = optimized_links(soup)

        naive_prompt = build_user_prompt(url, naive)
        optimized_prompt = build_user_prompt(url, optimized)

        naive_tok = count_tokens(encoding, LINK_SYSTEM_PROMPT, naive_prompt)
        optimized_tok = count_tokens(encoding, LINK_SYSTEM_PROMPT, optimized_prompt)

        row = {
            "url": url,
            "status": "ok",
            "naive_link_count": len(naive),
            "optimized_link_count": len(optimized),
            "links_removed": len(naive) - len(optimized),
            "naive_tiktoken": naive_tok,
            "optimized_tiktoken": optimized_tok,
            "tiktoken_saved": naive_tok - optimized_tok,
            "tiktoken_saved_pct": round(100 * (naive_tok - optimized_tok) / naive_tok, 1) if naive_tok else 0,
        }

        if live:
            try:
                naive_api = call_openai(client, LINK_SYSTEM_PROMPT, naive_prompt)
                optimized_api = call_openai(client, LINK_SYSTEM_PROMPT, optimized_prompt)
                row.update(
                    {
                        "naive_api_prompt_tokens": naive_api["prompt_tokens"],
                        "optimized_api_prompt_tokens": optimized_api["prompt_tokens"],
                        "api_prompt_tokens_saved": naive_api["prompt_tokens"] - optimized_api["prompt_tokens"],
                        "naive_api_completion_tokens": naive_api["completion_tokens"],
                        "optimized_api_completion_tokens": optimized_api["completion_tokens"],
                        "naive_latency_s": naive_api["latency_s"],
                        "optimized_latency_s": optimized_api["latency_s"],
                    }
                )
            except Exception as exc:
                print(f"  API call failed: {exc}")
                row["status"] = f"api_error: {exc}"

        rows.append(row)
        print(
            f"  links: {row.get('naive_link_count')} -> {row.get('optimized_link_count')} "
            f"| tiktoken: {row.get('naive_tiktoken')} -> {row.get('optimized_tiktoken')} "
            f"({row.get('tiktoken_saved_pct')}% saved)"
        )

    return rows


def summarize(rows):
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        print("\nNo successful rows to summarize.")
        return {}

    def total(key):
        return sum(r.get(key, 0) for r in ok_rows)

    n = len(ok_rows)
    summary = {
        "sites_ok": n,
        "sites_failed": len(rows) - n,
        "total_naive_links": total("naive_link_count"),
        "total_optimized_links": total("optimized_link_count"),
        "total_links_removed": total("links_removed"),
        "total_naive_tiktoken": total("naive_tiktoken"),
        "total_optimized_tiktoken": total("optimized_tiktoken"),
        "total_tiktoken_saved": total("tiktoken_saved"),
        "avg_tiktoken_saved_pct": round(sum(r.get("tiktoken_saved_pct", 0) for r in ok_rows) / n, 1),
    }

    if "naive_api_prompt_tokens" in ok_rows[0]:
        summary.update(
            {
                "total_naive_api_prompt_tokens": total("naive_api_prompt_tokens"),
                "total_optimized_api_prompt_tokens": total("optimized_api_prompt_tokens"),
                "total_api_prompt_tokens_saved": total("api_prompt_tokens_saved"),
                "avg_naive_latency_s": round(sum(r.get("naive_latency_s", 0) for r in ok_rows) / n, 3),
                "avg_optimized_latency_s": round(sum(r.get("optimized_latency_s", 0) for r in ok_rows) / n, 3),
            }
        )

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return summary


def save_results(rows, summary, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2)

    csv_path = out_dir / "results.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="only test the first N sites")
    parser.add_argument("--no-live", action="store_true", help="skip real OpenAI API calls, tiktoken only")
    args = parser.parse_args()

    results = run(limit=args.limit, live=not args.no_live)
    summary = summarize(results)
    save_results(results, summary, out_dir=Path(__file__).parent / "results")
