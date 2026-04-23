import hashlib

# simple in-memory store (replace with DB later)
API_KEYS = {
    "test_key_123": "basic_user"
}

def hash_key(key: str):
    return hashlib.sha256(key.encode()).hexdigest()

def validate_api_key(key: str):
    return key in API_KEYS