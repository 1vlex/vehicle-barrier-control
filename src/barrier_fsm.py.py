from collections import defaultdict
import time

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
