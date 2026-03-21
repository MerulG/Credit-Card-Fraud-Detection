import requests

BASE = "http://localhost:8000"

record = {
    "Time": 0, "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36,
    "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31,
    "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40,
    "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62,
}

print("--- /health ---")
print(requests.get(f"{BASE}/health").json())

print("\n--- /predict ---")
print(requests.post(f"{BASE}/predict", json=record).json())

print("\n--- /predict_batch (2 records) ---")
print(requests.post(f"{BASE}/predict_batch", json=[record, record]).json())
