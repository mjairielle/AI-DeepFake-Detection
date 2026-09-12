import sys
sys.path.append('.')
from detector import DeepfakeDetector

detector = DeepfakeDetector()
res = detector.analyze(r'C:\Users\Jai\.gemini\antigravity\brain\a4f2f2de-6df6-4429-b405-760ec1420604\.user_uploaded\media_1789236171754.png')
print(res)
