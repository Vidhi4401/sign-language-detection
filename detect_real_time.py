import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model


model = load_model("model/asl_model.h5")
with open("model/classes.json", "r") as f:
    classes = json.load(f)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

x1, y1, x2, y2 = 400, 100, 880, 580  # Larger ROI
img_size = 64
prediction_history = []
stable_prediction = ""
stable_threshold = 15

print("Press ESC to exit, C to clear prediction.")

while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        print("[ERROR] No frame captured.")
        continue

    h, w, _ = frame.shape
    if y2 > h or x2 > w:
        print(f"[ERROR] ROI {x1,y1,x2,y2} outside frame {w,h}")
        break

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        print("[WARNING] Empty ROI, skipping frame.")
        continue

    try:
        img = cv2.resize(roi, (img_size, img_size))
    except Exception as e:
        print("[ERROR] Resize failed:", e)
        continue

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    idx = np.argmax(pred)
    confidence = np.max(pred)
    label = classes[idx] if idx < len(classes) else "?"

    if confidence > 0.9:
        prediction_history.append(label)
        if len(prediction_history) > stable_threshold:
            prediction_history = prediction_history[-stable_threshold:]
            if all(p == prediction_history[0] for p in prediction_history):
                stable_prediction = prediction_history[0]
    else:
        prediction_history.clear()

    if stable_prediction:
        cv2.putText(frame, f"Letter: {stable_prediction}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow("ASL Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('c'):
        stable_prediction = ""
        prediction_history.clear()

cam.release()
cv2.destroyAllWindows()
