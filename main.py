import cv2
import numpy as np
from ultralytics import YOLO
import sys

# ================= PATH CONFIG (DO NOT CHANGE) =================
MODEL_PATH = "models/yolov8m.pt"
VIDEO_PATH = "videos/input.mp4"

SAFE_TH = 5
MOD_TH = 12

# ================= LOAD MODEL =================
model = YOLO(MODEL_PATH)

# ================= VIDEO =================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ ERROR: Cannot open video:", VIDEO_PATH)
    sys.exit(1)

# ================= GLOBALS =================
zones = []
mode = "rect"          # rect / free
drawing = False
current_pts = []
paused = False
delete_mode = False    # 🔑 NEW

# ================= RISK =================
def get_risk(count):
    if count <= SAFE_TH:
        return "SAFE", (0,255,0)
    elif count <= MOD_TH:
        return "MODERATE", (0,165,255)
    else:
        return "HIGH RISK", (0,0,255)

# ================= STATUS PANEL =================
def draw_status_panel(frame):
    overlay = frame.copy()
    panel_h = 90 + 40 * len(zones)

    cv2.rectangle(overlay, (0,0), (520, panel_h), (0,0,0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    cv2.putText(frame, "ZONE RISK STATUS",
                (20,45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (255,255,255), 3)

    for i, z in enumerate(zones):
        color = (0,255,0) if z["risk"]=="SAFE" else \
                (0,165,255) if z["risk"]=="MODERATE" else (0,0,255)

        cv2.putText(frame,
                    f"Zone {i+1}: {z['risk']} | Count: {z['count']}",
                    (20, 90 + i*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    color, 3)

    if delete_mode:
        cv2.putText(frame,
                    "DELETE MODE: Click inside a zone",
                    (550, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0,0,255), 3)
    return frame

# ================= DELETE ZONE =================
def delete_zone(point):
    global zones
    x, y = point
    for z in zones[:]:
        if z["type"] == "rect":
            (x1,y1),(x2,y2) = z["points"]
            if min(x1,x2)<=x<=max(x1,x2) and min(y1,y2)<=y<=max(y1,y2):
                zones.remove(z)
                return True
        else:
            if cv2.pointPolygonTest(np.array(z["points"]), (x,y), False) >= 0:
                zones.remove(z)
                return True
    return False

# ================= MOUSE =================
def mouse(event, x, y, flags, param):
    global drawing, current_pts, delete_mode

    if event == cv2.EVENT_LBUTTONDOWN:

        # 🔑 DELETE MODE
        if delete_mode:
            deleted = delete_zone((x,y))
            delete_mode = False   # auto-exit
            return

        # NORMAL MODES
        if mode == "rect":
            drawing = True
            current_pts = [(x,y)]

        elif mode == "free":
            current_pts.append((x,y))

    elif event == cv2.EVENT_MOUSEMOVE and drawing and mode == "rect":
        current_pts = [current_pts[0], (x,y)]

    elif event == cv2.EVENT_LBUTTONUP and drawing and mode == "rect":
        drawing = False
        zones.append({
            "type": "rect",
            "points": current_pts.copy(),
            "count": 0,
            "risk": "SAFE"
        })
        current_pts = []

# ================= WINDOW =================
cv2.namedWindow("Crowd Risk Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Crowd Risk Monitor", 1280, 720)
cv2.setMouseCallback("Crowd Risk Monitor", mouse)

# ================= MAIN LOOP =================
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        persons = []
        results = model(frame, stream=True)

        for r in results:
            for b in r.boxes:
                if int(b.cls[0]) == 0:
                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    persons.append((cx,cy))
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

        for z in zones:
            z["count"] = 0

        for cx,cy in persons:
            for z in zones:
                if z["type"] == "rect":
                    (x1,y1),(x2,y2) = z["points"]
                    if min(x1,x2)<=cx<=max(x1,x2) and min(y1,y2)<=cy<=max(y1,y2):
                        z["count"] += 1
                else:
                    if cv2.pointPolygonTest(np.array(z["points"]), (cx,cy), False) >= 0:
                        z["count"] += 1

        for z in zones:
            z["risk"], color = get_risk(z["count"])
            if z["type"] == "rect":
                cv2.rectangle(frame, z["points"][0], z["points"][1], color, 3)
            else:
                cv2.polylines(frame, [np.array(z["points"])], True, color, 3)

        if drawing and mode=="rect" and len(current_pts)==2:
            cv2.rectangle(frame, current_pts[0], current_pts[1], (255,255,255), 1)

        frame = draw_status_panel(frame)
        cv2.imshow("Crowd Risk Monitor", frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
    elif key == ord('r'):
        mode = "rect"
        drawing = False
        current_pts = []
    elif key == ord('f'):
        mode = "free"
        drawing = False
        current_pts = []
    elif key == 13 and mode=="free" and len(current_pts)>2:
        zones.append({
            "type": "free",
            "points": current_pts.copy(),
            "count": 0,
            "risk": "SAFE"
        })
        current_pts = []
        mode = "rect"
    elif key == ord('d'):
        delete_mode = True   # 🔑 ENTER DELETE MODE

cap.release()
cv2.destroyAllWindows()
