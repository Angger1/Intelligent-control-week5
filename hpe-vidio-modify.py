from ultralytics import YOLO
import cv2

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Buka video
video_path = "input_video.mp4"  # Ganti dengan path video yang ingin diproses
cap = cv2.VideoCapture(video_path)

# Dapatkan informasi video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Inisialisasi writer untuk menyimpan video output
output_path = "oWIN_20250317_20_43_21_Pro.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec untuk format MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Deteksi pose
    results = model(frame)
    
    # Tampilkan hasil
    for result in results:
        annotated_frame = result.plot()
    
    # Simpan frame yang telah dianotasi ke video output
    out.write(annotated_frame)
    
    # Tampilkan hasil (opsional)
    cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Proses selesai! Video hasil tersimpan di {output_path}")
