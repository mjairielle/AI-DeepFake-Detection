import sys
sys.path.append('.')
import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for p in [
    r'C:\Users\Jai\.gemini\antigravity\brain\a4f2f2de-6df6-4429-b405-760ec1420604\.user_uploaded\media_1789236171754.png',
    r'C:\Users\Jai\.gemini\antigravity\brain\a4f2f2de-6df6-4429-b405-760ec1420604\.user_uploaded\media_1789236210896.png'
]:
    gray = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    print(f"{p.split('.')[-2][-4:]} faces: {faces}")
