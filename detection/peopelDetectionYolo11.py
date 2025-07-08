import cv2
from ultralytics import YOLO
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
import os

#fungsi untuk menghitung waktu 
def format_elapsed_time(elapsed_seconds):
    """Format detik ke HH:MM:SS."""
    
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = int(elapsed_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

class Tracker:
    def __init__(self):
        self.object_tracker = DeepSort( 
            max_age=30, #Jumlah frame maksimal tanpa deteksi ulang sebelum ID objek dianggap hilang.
            n_init=5, #Jumlah deteksi berurutan yang dibutuhkan agar sebuah track dianggap valid/confirmed.
            nms_max_overlap=0.7, # antar bounding box untuk Non-Maximum Suppression (NMS).
            max_cosine_distance=0.3, # Ambang batas maksimum jarak cosine antar fitur vektor (embedding) untuk mengaitkan deteksi baru ke track lama.
            nn_budget=None,
           # override_track_class=None,
            embedder="mobilenet", #model untuk ekstraksi fitur (embedding) dari crop bounding box. Biasanya digunakan untuk membantu membedakan antar objek yang mirip.
            half=True,
            #bgr=True,
            #embedder_model_name=None,
           # embedder_wts=None,
           # today=None,
        )

    def track(self, detections, frame):
        tracks = self.object_tracker.update_tracks(detections, frame=frame)
        matched = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            # Ambil bounding box (x1, y1, x2, y2)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # Hitung lebar dan tinggi
            w, h = x2 - x1, y2 - y1
            matched.append((track.track_id, [x1, y1, w, h]))

        return matched
    
class YoloDetector:
    # membuat model path dan membuat treshold 
    def __init__(self, model_path, confidence):
        self.model = YOLO(model_path)
        self.classList = ["person"]
        self.confidence = confidence

    def detect(self, image):
        results = self.model.predict(image, conf=self.confidence, iou=0.5)
        result = results[0]
        detections = self.make_detections(result)
        return detections

    def make_detections(self, result):
        #mengambil bounding box hasil deteksi yang di simpan pada variabel box
        boxes = result.boxes
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            class_number = int(box.cls[0])
            if result.names[class_number] not in self.classList:
                continue
            conf = float(box.conf[0])
            detections.append(([x1, y1, w, h], class_number, conf))
        return detections

def main():
    #model path yang dipakai serta confiden
    detector = YoloDetector(
        model_path="yolo11n.pt",
        confidence=0.3
    )
    tracker = Tracker()

# input camera yang dipakai
    rtsp_url = "rtsp://admin:Admin123@192.168.28.30:554/Streaming/Channels/101"
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    if not cap.isOpened():
        print("Error: Camera is trouble")
        exit()

#membuat directory untuk menyimpan hasil 
    #os.makedirs("videos", exist_ok=True)

    #frame_width = int(cap.get(3))
    #frame_height = int(cap.get(4))

    #now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
   # output_filename = f"videos/output_{now_str}.mp4"

    #out = cv2.VideoWriter(
        #output_filename,
       # cv2.VideoWriter_fourcc(*'mp4v'),
        #20.0,
      #  (frame_width, frame_height)
   # )
# tracke untuk menghitung waktu object terdeteksi 
    track_start_times = dict()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = datetime.now()
        start_time = time.perf_counter()
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #ubah frame ke RGB
        detections = detector.detect(frame_RGB)
        matched = tracker.track(detections, frame)

        # matched = [(track_id, [x, y, w, h])]
        for track_id, bbox in matched:
            x, y, w, h = bbox
            #menyimpan waktu tracking id
            if track_id not in track_start_times:
                track_start_times[track_id] = now
            elapsed_seconds = (now - track_start_times[track_id]).total_seconds() #hitung lama object terdeteksi 
            elapsed_str = format_elapsed_time(elapsed_seconds)

            # memberi label text pada bonding box 
            label = f"ID={track_id} , {elapsed_str}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)
        print(f"Current fps: {fps:.2f}")

        #out.write(frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    #out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
