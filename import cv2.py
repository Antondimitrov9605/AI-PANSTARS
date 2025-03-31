import cv2
import numpy as np
import os

# Зареждане на последователни изображения
image_folder = 'astro_images/'
images = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
prev_frame = None

for filename in images:
    path = os.path.join(image_folder, filename)
    frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if prev_frame is None:
        prev_frame = frame
        continue

    # Откриване на разлика между кадри (астероиди се движат, звездите не)
    diff = cv2.absdiff(prev_frame, frame)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Намиране на обекти (контури) в диференцираното изображение
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"{filename}: Открити {len(contours)} потенциални движещи се обекта")

    for cnt in contours:
        if cv2.contourArea(cnt) > 5:  # Праг за шум
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 1)
    
    cv2.imshow("Detected", frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    prev_frame = frame

cv2.destroyAllWindows()
