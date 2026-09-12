"""Quick test: analyze -> feedback -> learning status."""
import numpy as np
import cv2
import requests
import json
import os

BASE = "http://127.0.0.1:5000"

# 1. Check learning status before
print("=== LEARNING STATUS (before) ===")
r = requests.get(f"{BASE}/api/learning/status")
print(json.dumps(r.json(), indent=2))

# 2. Analyze a synthetic image
img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.circle(img, (150, 130), 80, (200, 180, 160), -1)
cv2.imwrite("test_synth.jpg", img)

with open("test_synth.jpg", "rb") as f:
    r = requests.post(f"{BASE}/api/analyze", files={"file": ("synth.jpg", f)})
result = r.json()
print("\n=== ANALYSIS RESULT ===")
print("Verdict:", result["verdict"], " Prob:", result["deepfake_probability"])

# 3. Submit feedback: user says this IS a deepfake
feedback = {
    "file_hash": result["file_hash"],
    "media_type": result["media_type"],
    "original_verdict": result["verdict"],
    "corrected_verdict": "deepfake",
    "module_scores": result["module_scores"],
    "flags": result["flags"],
    "manipulation_type": "full_synthesis",
}
r = requests.post(f"{BASE}/api/feedback", json=feedback)
print("\n=== FEEDBACK RESPONSE ===")
print(json.dumps(r.json(), indent=2))

# 4. Submit more feedback to trigger classifier training
for i in range(12):
    scores = {k: v + np.random.uniform(-0.1, 0.1)
              for k, v in result["module_scores"].items()}
    fb = {
        "file_hash": f"fake_hash_{i}",
        "media_type": "image",
        "original_verdict": "suspicious",
        "corrected_verdict": "deepfake" if i % 3 != 0 else "authentic",
        "module_scores": scores,
        "flags": ["test"],
    }
    requests.post(f"{BASE}/api/feedback", json=fb)

# 5. Check learning status after
print("\n=== LEARNING STATUS (after 13 feedbacks) ===")
r = requests.get(f"{BASE}/api/learning/status")
print(json.dumps(r.json(), indent=2))

# 6. Re-analyze same image to see adaptive effect
with open("test_synth.jpg", "rb") as f:
    r = requests.post(f"{BASE}/api/analyze", files={"file": ("synth.jpg", f)})
result2 = r.json()
print("\n=== RE-ANALYSIS (after learning) ===")
print("Verdict:", result2["verdict"], " Prob:", result2["deepfake_probability"])

os.remove("test_synth.jpg")
print("\nDone!")
