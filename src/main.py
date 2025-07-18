import cv2
import time
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

class BarrierFSM:
    def __init__(self):
        self.STATE_IDLE = "IDLE"
        self.STATE_OPENING = "OPENING"
        self.STATE_OPEN = "OPEN"
        self.STATE_CLOSING = "CLOSING"
        self.state = self.STATE_IDLE
        self.target_id = None
        self.downward_time = defaultdict(float)
        self.disappear_time = 0
        self.last_activity_time = time.time()

        # Параметры
        self.OPEN_THRESH = 1    # секунд спуска
        self.CLOSE_DELAY = 5     # сек
        self.INACTIVITY_TIMEOUT = 120  # сек
        self.OPEN_CLOSE_DUR = 1   # сек
        self.MIN_ANGLE = 30       # град

    def update(self, current_time, tracked_objects, dt):
        # обновление активности
        if tracked_objects:
            self.last_activity_time = current_time

        # таймаут бездействия
        if current_time - self.last_activity_time >= self.INACTIVITY_TIMEOUT:
            if self.state not in [self.STATE_IDLE, self.STATE_CLOSING]:
                self.state = self.STATE_CLOSING
                self.trigger_time = current_time
                print(f"[{current_time:.2f}] Принудительное закрытие по таймауту бездействия")

        if self.state == self.STATE_IDLE:
            for obj_id, _, movement_y, angle in tracked_objects:
                if movement_y > 0 and abs(angle) >= self.MIN_ANGLE:
                    self.downward_time[obj_id] += dt
                    if self.downward_time[obj_id] >= self.OPEN_THRESH:
                        self.state = self.STATE_OPENING
                        self.target_id = obj_id
                        self.trigger_time = current_time
                        print(f"[{current_time:.2f}] Открытие по длительному спуску объекта {obj_id}")
                else:
                    self.downward_time[obj_id] = 0

        elif self.state == self.STATE_OPENING:
            if current_time - self.trigger_time >= self.OPEN_CLOSE_DUR:
                self.state = self.STATE_OPEN
                print(f"[{current_time:.2f}] Шлагбаум открыт")

        elif self.state == self.STATE_OPEN:
            target_found = any(obj_id == self.target_id for obj_id, *_ in tracked_objects)
            if not target_found:
                if self.disappear_time == 0:
                    self.disappear_time = current_time
                elif current_time - self.disappear_time >= self.CLOSE_DELAY:
                    self.state = self.STATE_CLOSING
                    self.trigger_time = current_time
                    print(f"[{current_time:.2f}] Открытие завершено, начинаем закрытие")
            else:
                self.disappear_time = 0

        elif self.state == self.STATE_CLOSING:
            if current_time - self.trigger_time >= self.OPEN_CLOSE_DUR:
                self.state = self.STATE_IDLE
                self.target_id = None
                self.downward_time.clear()
                self.disappear_time = 0
                print(f"[{current_time:.2f}] Шлагбаум закрыт")


def calculate_movement_angle(track_history, track_id, current_point):
    if track_id not in track_history or len(track_history[track_id]) < 2:
        return 0
    prev_point = track_history[track_id][-1]
    dx = current_point[0] - prev_point[0]
    dy = current_point[1] - prev_point[1]
    if dx == 0:
        return 90.0 if dy > 0 else -90.0
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    vertical_angle = 90.0 - abs(angle_deg)
    return vertical_angle if dy > 0 else -vertical_angle


def process_video(video_path):
    model = YOLO("runs/detect/vehicles_detection/weights/best.pt")
    barrier = BarrierFSM()
    cap = cv2.VideoCapture(video_path)
    track_history = defaultdict(lambda: deque(maxlen=30))
    prev_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        h = frame.shape[0]

        results = model.track(frame, persist=True, classes=[0], conf=0.4, verbose=False)
        tracked_objects = []
        forced_open = False

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            ids = results[0].boxes.id.int().cpu().tolist()
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = box
                cx, cy = float((x1+x2)/2), float((y1+y2)/2)

                # принудительное открытие по линии 1/3
                if cy > h/3 and barrier.state == barrier.STATE_IDLE:
                    barrier.state = barrier.STATE_OPENING
                    barrier.trigger_time = current_time
                    barrier.target_id = tid
                    forced_open = True
                    print(f"[{current_time:.2f}] Принудительное открытие: объект {tid} ниже 1/3")

                prev_pt = track_history[tid][-1] if track_history[tid] else None
                movement_y = cy - prev_pt[1] if prev_pt is not None else 0
                track_history[tid].append((cx, cy))
                angle = calculate_movement_angle(track_history, tid, (cx, cy))
                tracked_objects.append((tid, cy, movement_y, angle))

                # отрисовка
                col = (0, 255, 0)
                if tid == barrier.target_id:
                    col = (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                for i in range(1, len(track_history[tid])):
                    p1 = track_history[tid][i-1]; p2 = track_history[tid][i]
                    cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), col, 2)
                cv2.putText(frame, f"ID:{tid} A:{angle:.1f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        if not forced_open:
            barrier.update(current_time, tracked_objects, dt)

        # отображение
        state = barrier.state
        status = "OPEN" if state in [barrier.STATE_OPENING, barrier.STATE_OPEN] else "CLOSED"
        color = (0,255,0) if status=="OPEN" else (0,0,255)
        cv2.putText(frame, f"State: {state}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Barrier: {status}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.line(frame, (0, h//3), (frame.shape[1], h//3), (255,0,0), 2)

        cv2.imshow("Barrier Control System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    process_video(args.video)
