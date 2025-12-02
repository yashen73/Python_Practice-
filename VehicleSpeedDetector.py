import cv2
import time
from ultralytics import YOLO

# ---------------------------
# CONFIGURATION
# ---------------------------
AREA_TOP = 200           # Y-position for entry
AREA_BOTTOM = 350        # Y-position for exit
REAL_DISTANCE_M = 10     # Meters between the two lines

model = YOLO("yolov8n.pt")  # or yolov11n.pt if you have it

cap = cv2.VideoCapture("Road_Traffic.mp4")

# Track object entry/exit times
vehicle_times = {}
vehicle_speeds = {}  # store calculated speeds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            if cls not in [2, 3, 5, 7]:
                # car, motorcycle, bus, truck
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0])

            cy = int((y1 + y2) / 2)  # center point Y

            # ------------------------
            # ENTRY LINE CROSS
            # ------------------------
            if AREA_TOP - 5 < cy < AREA_TOP + 5:
                if track_id not in vehicle_times:
                    vehicle_times[track_id] = [time.time(), None]

            # ------------------------
            # EXIT LINE CROSS
            # ------------------------
            if AREA_BOTTOM - 5 < cy < AREA_BOTTOM + 5:
                if track_id in vehicle_times and vehicle_times[track_id][1] is None:
                    vehicle_times[track_id][1] = time.time()

                    # Calculate speed
                    t1, t2 = vehicle_times[track_id]
                    travel_time = t2 - t1

                    if travel_time > 0:
                        mps = REAL_DISTANCE_M / travel_time
                        kmph = mps * 3.6
                        vehicle_speeds[track_id] = kmph

            # ------------------------
            # DRAW BOUNDING BOX
            # ------------------------
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ------------------------
            # LABEL SPEED ON BOX
            # ------------------------
            speed_text = ""

            if track_id in vehicle_speeds:
                speed_text = f"{vehicle_speeds[track_id]:.1f} km/h"
            else:
                speed_text = "Calculating..."

            cv2.putText(frame, f"ID {track_id} | {speed_text}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

    # Draw the measurement lines
    cv2.line(frame, (0, AREA_TOP), (frame.shape[1], AREA_TOP), (255, 0, 0), 2)
    cv2.line(frame, (0, AREA_BOTTOM), (frame.shape[1], AREA_BOTTOM), (0, 0, 255), 2)

    cv2.imshow("Vehicle Speed", frame)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
