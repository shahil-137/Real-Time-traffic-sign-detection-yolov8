

# from flask import Flask, render_template, Response, jsonify
# from ultralytics import YOLO
# import cv2
# import pyttsx3
# import threading
# import time
# from datetime import datetime

# app = Flask(__name__)

# # -------------------- CLASS MEANINGS --------------------
# CLASS_MEANINGS = {
#     "pn": "No Parking",
#     "pne": "No Entry",
#     "i5": "Keep Right",
#     "i4": "Motor Vehicles Only",
#     "i2": "Bicycles Only",
#     "i2r": "Bicycles - Keep Right",
#     "i4l": "Motor Vehicles Keep Left",
#     "pl5": "Speed Limit 5",
#     "pl30": "Speed Limit 30",
#     "pl40": "Speed Limit 40",
#     "pl50": "Speed Limit 50",
#     "pl60": "Speed Limit 60",
#     "pl80": "Speed Limit 80",
#     "pl100": "Speed Limit 100",
#     "pl120": "Speed Limit 120",
#     "il60": "Minimum Speed 60",
#     "il80": "Minimum Speed 80",
#     "p5": "No U Turn",
#     "p10": "No Motor Cars",
#     "p11": "No Horn",
#     "p13": "No Motor Vehicles",
#     "p23": "No Left Turn",
#     "p26": "No Trucks",
#     "ip": "Pedestrian Crossing",
#     "w57": "Pedestrian Crossing Ahead"
# }

# # -------------------- CONFIG --------------------
# DETECT_CONF = 0.45
# VOICE_CONF = 0.60
# STABLE_FRAMES_REQ = 4

# # -------------------- STATS --------------------
# stats = {
#     "fps": 0.0,
#     "total_detections": 0,
#     "top_conf": 0.0,
#     "recent_detections": []
# }

# detection_id = 0
# camera_active = False

# # -------------------- VOICE SYSTEM --------------------
# active_voice_threads = {}
# stable_counter = {}

# def voice_loop_worker(label, stop_event):

#     engine = pyttsx3.init()
#     engine.setProperty("rate",175)

#     while not stop_event.is_set():

#         engine.say(label)
#         engine.runAndWait()

#     engine.stop()

# def start_voice_loop(label):

#     if label in active_voice_threads:
#         return

#     stop_event = threading.Event()

#     thread = threading.Thread(
#         target=voice_loop_worker,
#         args=(label,stop_event),
#         daemon=True
#     )

#     active_voice_threads[label] = {"thread":thread,"stop_flag":stop_event}

#     thread.start()

# def stop_voice_loop(label):

#     if label not in active_voice_threads:
#         return

#     active_voice_threads[label]["stop_flag"].set()
#     del active_voice_threads[label]

# def update_stability(detected_labels):

#     all_known = set(stable_counter.keys()) | detected_labels

#     for label in all_known:

#         if label in detected_labels:
#             stable_counter[label] = stable_counter.get(label,0) + 1
#         else:
#             stable_counter[label] = 0

#     return {l for l,c in stable_counter.items() if c >= STABLE_FRAMES_REQ}

# # -------------------- MODEL --------------------
# model = YOLO("runs/detect/traffic_sign_25cls1024_finetune25ep/weights/best.pt")
# cap = None

# # -------------------- VIDEO STREAM --------------------
# def generate_frames():

#     global cap, camera_active, stats, detection_id

#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Cannot open webcam")
#         return

#     prev_time = time.time()

#     while camera_active:

#         ret, frame = cap.read()

#         if not ret:
#             break

#         results = model(frame, conf=DETECT_CONF, imgsz=1024)
#         r = results[0]

#         best_box = None
#         best_conf = 0

#         for box in r.boxes:

#             conf = float(box.conf[0])

#             if conf > best_conf:
#                 best_conf = conf
#                 best_box = box

#         detected_labels_this_frame = set()
#         high_conf_signs = set()

#         if best_box is not None and best_conf >= VOICE_CONF:

#             cls_id = int(best_box.cls[0])
#             class_code = model.names[cls_id]

#             label = CLASS_MEANINGS.get(class_code,class_code)

#             x1,y1,x2,y2 = map(int,best_box.xyxy[0])

#             # Only ONE GREEN detection
#             cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,100),2)

#             cv2.putText(
#                 frame,
#                 f"{label} ({best_conf:.2f})",
#                 (x1,y1-10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (0,255,100),
#                 2
#             )

#             detected_labels_this_frame.add(label)
#             high_conf_signs.add(label)

#             detection_id += 1
#             stats["total_detections"] += 1
#             stats["top_conf"] = best_conf

#             entry = {
#                 "id": detection_id,
#                 "label": label,
#                 "conf": round(best_conf,3),
#                 "time": datetime.now().strftime("%H:%M:%S")
#             }

#             stats["recent_detections"].insert(0,entry)
#             stats["recent_detections"] = stats["recent_detections"][:20]

#         # Stability logic
#         stable_labels = update_stability(detected_labels_this_frame)

#         for label in stable_labels:
#             start_voice_loop(label)

#         for label in set(active_voice_threads.keys()) - high_conf_signs:
#             stop_voice_loop(label)

#         # FPS
#         curr_time = time.time()
#         fps = 1.0 / max(curr_time - prev_time,1e-6)
#         prev_time = curr_time

#         stats["fps"] = round(fps,1)

#         cv2.putText(frame,f"FPS: {fps:.1f}",(10,30),
#                     cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

#         ret2,buffer = cv2.imencode('.jpg',frame)

#         if not ret2:
#             continue

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' +
#                buffer.tobytes() +
#                b'\r\n')

#     if cap:
#         cap.release()

# # -------------------- ROUTES --------------------

# @app.route('/')
# def landing():
#     return render_template('landing.html')

# @app.route('/dashboard')
# def dashboard():
#     return render_template('index.html')

# @app.route('/video')
# def video():

#     global camera_active

#     camera_active = True

#     return Response(
#         generate_frames(),
#         mimetype='multipart/x-mixed-replace; boundary=frame'
#     )

# @app.route('/stop_camera')
# def stop_camera():

#     global camera_active

#     camera_active = False

#     return jsonify({"status":"stopped"})

# @app.route('/stats')
# def get_stats():
#     return jsonify(stats)

# # -------------------- RUN --------------------
# if __name__ == "__main__":

#     print("Traffic Sign Detection Flask App Starting...")

#     app.run(debug=False, threaded=True)


from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import pyttsx3
import threading
import time
from datetime import datetime

app = Flask(__name__)

# -------------------- CLASS MEANINGS --------------------
CLASS_MEANINGS = {
    "pn": "No Parking",
    "pne": "No Entry",
    "i5": "Keep Right",
    "i4": "Motor Vehicles Only",
    "i2": "Bicycles Only",
    "i2r": "Bicycles - Keep Right",
    "i4l": "Motor Vehicles Keep Left",
    "pl5": "Speed Limit 5",
    "pl30": "Speed Limit 30",
    "pl40": "Speed Limit 40",
    "pl50": "Speed Limit 50",
    "pl60": "Speed Limit 60",
    "pl80": "Speed Limit 80",
    "pl100": "Speed Limit 100",
    "pl120": "Speed Limit 120",
    "il60": "Minimum Speed 60",
    "il80": "Minimum Speed 80",
    "p5": "No U Turn",
    "p10": "No Motor Cars",
    "p11": "No Horn",
    "p13": "No Motor Vehicles",
    "p23": "No Left Turn",
    "p26": "No Trucks",
    "ip": "Pedestrian Crossing",
    "w57": "Pedestrian Crossing Ahead"
}

# -------------------- CONFIDENCE --------------------
DETECT_CONF = 0.45
VOICE_CONF = 0.75
LOOP_INTERVAL = 2.0

# -------------------- STATS --------------------
stats = {
    "fps": 0.0,
    "total_detections": 0,
    "top_conf": 0.0,
    "recent_detections": []
}

camera_active = False
detection_id = 0

# -------------------- VOICE SYSTEM --------------------
active_voice_threads = {}

def voice_loop_worker(label, stop_event):

    engine = pyttsx3.init()
    engine.setProperty("rate", 175)

    while not stop_event.is_set():

        print(f"🔊 Speaking: {label}")
        engine.say(label)
        engine.runAndWait()
        stop_event.wait(LOOP_INTERVAL)

    engine.stop()

def start_voice_loop(label):

    if label in active_voice_threads:
        return

    stop_event = threading.Event()

    thread = threading.Thread(
        target=voice_loop_worker,
        args=(label, stop_event),
        daemon=True
    )

    active_voice_threads[label] = {
        "thread": thread,
        "stop_flag": stop_event
    }

    thread.start()

def stop_voice_loop(label):

    if label not in active_voice_threads:
        return

    active_voice_threads[label]["stop_flag"].set()
    del active_voice_threads[label]

# -------------------- MODEL --------------------
model = YOLO("runs/detect/traffic_sign_gap_reduction_final/weights/best.pt")

cap = None

# -------------------- VIDEO STREAM --------------------
def generate_frames():

    global cap, camera_active, stats, detection_id

    cap = cv2.VideoCapture(0)

    prev_time = time.time()

    while camera_active:

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=DETECT_CONF, imgsz=1024)
        r = results[0]

        high_conf_signs = set()

        for box in r.boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            class_code = model.names[cls_id]
            label = CLASS_MEANINGS.get(class_code, class_code)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            cv2.putText(
                frame,
                f"{label} ({conf:.2f})",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

            # Voice trigger
            if conf >= VOICE_CONF:

                high_conf_signs.add(label)
                start_voice_loop(label)

                detection_id += 1
                stats["total_detections"] += 1
                stats["top_conf"] = conf

                entry = {
                    "id": detection_id,
                    "label": label,
                    "conf": round(conf,3),
                    "time": datetime.now().strftime("%H:%M:%S")
                }

                stats["recent_detections"].insert(0, entry)
                stats["recent_detections"] = stats["recent_detections"][:20]

        # Stop voice if sign disappears
        for label in set(active_voice_threads.keys()) - high_conf_signs:
            stop_voice_loop(label)

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        stats["fps"] = round(fps,1)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() +
               b'\r\n')

    if cap:
        cap.release()

# -------------------- ROUTES --------------------
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/video')
def video():

    global camera_active
    camera_active = True

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stop_camera')
def stop_camera():

    global camera_active
    camera_active = False

    return jsonify({"status":"stopped"})

@app.route('/stats')
def get_stats():
    return jsonify(stats)

# -------------------- RUN --------------------
if __name__ == "__main__":

    print("Traffic Sign Detection Flask App Starting...")

    app.run(debug=False, threaded=True)