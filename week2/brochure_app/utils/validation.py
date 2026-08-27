def validate_api_key(api_key, api_key_name):
    if not api_key:
        raise ValueError("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
    elif not api_key.startswith("sk-proj-") and not api_key.startswith("AIz") and not api_key.startswith("AQ."):
        raise ValueError(f"An API key was found, but it doesn't start {api_key_name}; please check you're using the right key - see troubleshooting notebook")
    return True