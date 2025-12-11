import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# Load model
model = load_model("model/asl_model.keras")

# Load class names
with open("model/classes.json", "r") as f:
    classes = json.load(f)

img_size = 128

# Camera setup
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ROI for the hand
x1, y1 = 450, 80
x2, y2 = 880, 580
# ROI for the hand x1, y1 = 400, 100 x2, y2 = 880, 580

# Stability buffer
history = []
stable_letter = ""
stable_frames = 10 # good balance

print("Press ESC to exit. Press C to clear.")

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    roi = frame[y1:y2, x1:x2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    # # Remove background using thresholding
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35,35), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # # Convert to 3 channels so the model can read it
    roi_clean = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(roi, (img_size, img_size))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    idx = np.argmax(pred)
    confidence = pred[0][idx]

    label = classes[idx]

    if confidence > 0.4:
        history.append(label)
        if len(history) > stable_frames:
            history = history[-stable_frames:]
            if all(h == history[0] for h in history):
                stable_letter = history[0]
    else:
        history.clear()

    # Draw UI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 255), 2)

    if stable_letter:
        cv2.putText(frame, f"{stable_letter} ({confidence:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("ASL Detection", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == ord('c'):
        stable_letter = ""
        history.clear()

cam.release()
cv2.destroyAllWindows()
