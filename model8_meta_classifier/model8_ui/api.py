import requests

API_URL = "http://127.0.0.1:8000"

def fetch(path):
    r = requests.get(f"{API_URL}{path}")
    r.raise_for_status()
    return r.json()

def get_market_status():
    return fetch("/market/status")

def get_performance():
    return fetch("/metrics/performance")
