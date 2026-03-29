import threading
import websocket
import json
from datetime import datetime

latest_tick = {}

def _on_message(ws, message):
    global latest_tick
    data = json.loads(message)
    latest_tick = {
        "price": float(data["p"]),
        "time": datetime.utcnow()
    }

def _on_error(ws, error):
    pass

def _on_close(ws):
    pass

def _on_open(ws):
    pass

def run_ws_background():
    def _run():
        ws = websocket.WebSocketApp(
            "wss://stream.binance.com:9443/ws/btcusdt@trade",
            on_message=_on_message,
            on_error=_on_error,
            on_close=_on_close,
            on_open=_on_open
        )
        ws.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
