from ultralytics import YOLO
import cv2
import mediapipe as mp

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Balikkan frame agar mirror
    frame = cv2.flip(frame, 1)
    
    # Deteksi pose tubuh dengan YOLOv8
    results = model(frame)  
    annotated_frame = results[0].plot()

    # Konversi warna untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    # Jika ada tangan terdeteksi
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Tampilkan hasil pada jendela OpenCV
    cv2.imshow("YOLOv8 Pose + Hand Detection", annotated_frame)

    # Tombol 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilis kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
