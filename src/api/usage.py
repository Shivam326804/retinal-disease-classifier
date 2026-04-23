USAGE_DB = {}

def log_usage(api_key: str):
    if api_key not in USAGE_DB:
        USAGE_DB[api_key] = 0
    USAGE_DB[api_key] += 1

def get_usage(api_key: str):
    return USAGE_DB.get(api_key, 0)