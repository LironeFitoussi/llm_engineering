import sys
from utils.ai_functions import stream_brochure


def main():
    company_name = sys.argv[1] if len(sys.argv) > 1 else input("Enter the company name: ").strip()
    url = sys.argv[2] if len(sys.argv) > 2 else input("Enter the company's website URL: ").strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url

    stream_brochure(company_name, url)


if __name__ == "__main__":
    main()
