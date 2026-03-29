from fastapi import FastAPI
from .inference import latest_state, history

app = FastAPI(title="Bitcoin Market Intelligence — Model 8")

@app.get("/market/status")
def market_status():
    return latest_state()

@app.get("/signal/history")
def signal_history():
    return history()
