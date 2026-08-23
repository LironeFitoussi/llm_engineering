def validate_api_key(api_key: str) -> None:
    if not api_key:
        raise SystemExit("No API key was found - please check your .env file")
    if not api_key.startswith("sk-proj-"):
        raise SystemExit("API key doesn't start with sk-proj-; please check you're using the right key")
    if api_key.strip() != api_key:
        raise SystemExit("API key has space or tab characters at the start or end - please remove them")
    return True