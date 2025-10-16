import cv2
import torch
import numpy as np
from collections import defaultdict
import logging
from typing import Dict, Any
from ultralytics import YOLO

class AccurateGPUAggressiveDrivingDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.logger.info(f"Using device: {device}")
        self.model = YOLO("yolo11n.pt")
        self.model.to(self.device)
        self.model.eval()
        self.stats = {
            "total_count": 0,
            "aggressive_count": 0,
            "acc_count": 0,
            "br_count": 0,
            "lane_count": 0,
            "avg_speed": 0,
            "avg_acc": 0,
            "avg_angle": 0
        }

        self.helping_data = {
            "sum_speed": 0,
            "sum_acc": 0,
            "sum_angle": 0,
            "count_speed": 0,
            "count_acc": 0,
            "count_angle": 0
        }

        self.track_history = defaultdict(lambda: {
            'positions': torch.zeros((120, 2), device=self.device),
            'timestamps': torch.zeros(120, device=self.device),
            'count': 0,
            'aggressive_count': 0,
            'last_aggressive_frame': 0,
            'world_speed': torch.tensor(0.0, device=self.device),
            'world_acceleration': torch.tensor(0.0, device=self.device),
            'movement_angle': torch.tensor(0.0, device=self.device),
            'speed_history': torch.zeros(30, device=self.device),
            'speed_history_count': 0,
            'is_calibrated': False,
            'pixel_positions': torch.zeros((120, 2), device=self.device)
        })

        # Консервативные пороги
        self.ACCELERATION_THRESHOLD = torch.tensor(20.0, device=self.device)
        self.DECELERATION_THRESHOLD = torch.tensor(-15.0, device=self.device)
        self.LANE_CHANGE_ANGLE_THRESHOLD = torch.tensor(15.0, device=self.device)
        self.MIN_TRACK_LENGTH = 10
        self.MIN_CALIBRATION_FRAMES = 3

        # Параметры калибровки перспективы
        self.perspective_matrix = None
        self.inverse_perspective_matrix = None
        self.is_perspective_calibrated = False
        self.calibration_points = []
        self.calibration_complete = False
        self.real_world_size = None
        self.meters_per_pixel_x = 50
        self.meters_per_pixel_y = 150

        self.calibration_data = {
            'frame_size': None,
            'scale_factor': 1.0
        }
        # self.set_calibration(calibration_data=self.calibration_data)

    def set_calibration(self, calibration_data: Dict[str, Any]):
        """Установка калибровочных параметров"""
        self.perspective_matrix = np.array(calibration_data['perspective_matrix'])
        self.inverse_perspective_matrix = np.array(calibration_data['inverse_perspective_matrix'])
        self.real_world_size = calibration_data['real_world_size']
        self.meters_per_pixel_x = calibration_data['meters_per_pixel_x']
        self.meters_per_pixel_y = calibration_data['meters_per_pixel_y']
        self.is_perspective_calibrated = True
        self.logger.info("Perspective calibration loaded successfully")

    def pixel_to_world(self, pixel_coords):
        """Преобразование пиксельных координат в мировые"""
        if not self.is_perspective_calibrated:
            return torch.tensor([pixel_coords[0], pixel_coords[1]], device=self.device)

        pixel_points = np.array([[pixel_coords[0], pixel_coords[1]]], dtype=np.float32)
        world_points_pixels = cv2.perspectiveTransform(
            pixel_points.reshape(1, -1, 2), self.perspective_matrix
        )

        world_x = world_points_pixels[0, 0, 0] * self.meters_per_pixel_x
        world_y = world_points_pixels[0, 0, 1] * self.meters_per_pixel_y

        return torch.tensor([world_x, world_y], device=self.device)

    def update_track_history(self, track_id, pixel_position, current_time):
        """Обновление истории треков"""
        track_data = self.track_history[track_id]

        if track_data['count'] < 120:
            track_data['pixel_positions'][track_data['count']] = pixel_position
            track_data['timestamps'][track_data['count']] = current_time

            if self.is_perspective_calibrated:
                world_position = self.pixel_to_world(pixel_position.cpu().numpy())
                track_data['positions'][track_data['count']] = world_position

            track_data['count'] += 1
        else:
            track_data['pixel_positions'] = torch.roll(track_data['pixel_positions'], -1, 0)
            track_data['timestamps'] = torch.roll(track_data['timestamps'], -1, 0)
            track_data['pixel_positions'][-1] = pixel_position
            track_data['timestamps'][-1] = current_time

            if self.is_perspective_calibrated:
                world_position = self.pixel_to_world(pixel_position.cpu().numpy())
                track_data['positions'] = torch.roll(track_data['positions'], -1, 0)
                track_data['positions'][-1] = world_position

        # Калибровка скорости
        if (track_data['count'] >= self.MIN_CALIBRATION_FRAMES and
                not track_data['is_calibrated'] and
                self.is_perspective_calibrated):
            self._calibrate_track_speed(track_id)

    def _calibrate_track_speed(self, track_id):
        """Калибровка скорости для трека"""
        track_data = self.track_history[track_id]

        if track_data['count'] < 10:
            return

        valid_positions = track_data['positions'][:track_data['count']]
        valid_timestamps = track_data['timestamps'][:track_data['count']]

        speeds = []
        for i in range(1, min(20, track_data['count'])):
            idx1 = track_data['count'] - i
            idx2 = track_data['count'] - i - 1

            if idx2 < 0:
                break
            pos1 = valid_positions[idx1]
            pos2 = valid_positions[idx2]
            time1 = valid_timestamps[idx1]
            time2 = valid_timestamps[idx2]

            if time1 > time2:
                displacement = torch.norm(pos1 - pos2)
                time_delta = time1 - time2

                if time_delta > 0.001:
                    speed = displacement / time_delta
                    speeds.append(speed)

        if speeds:
            speeds_tensor = torch.stack(speeds)
            median_speed = torch.median(speeds_tensor)
            track_data['world_speed'] = median_speed
            track_data['is_calibrated'] = True
            track_data['speed_history'][0] = median_speed
            self.helping_data['sum_speed'] += median_speed
            self.helping_data["count_speed"] += 1
            self.stats["avg_speed"] = self.helping_data['sum_speed'] / self.helping_data["count"]
            track_data['speed_history_count'] = 1

    def calculate_world_metrics(self, track_data):
        """Расчет метрик в мировых координатах"""
        if track_data['count'] < 10 or not track_data['is_calibrated']:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        valid_positions = track_data['positions'][:track_data['count']]
        valid_timestamps = track_data['timestamps'][:track_data['count']]

        window_size = min(10, track_data['count'])
        recent_positions = valid_positions[-window_size:]
        recent_timestamps = valid_timestamps[-window_size:]

        if len(recent_positions) >= 5:
            t_relative = recent_timestamps - recent_timestamps[0]

            # Для X координаты
            x_coords = recent_positions[:, 0]
            A_x = torch.stack([t_relative, torch.ones_like(t_relative)], dim=1)
            try:
                coefficients_x = torch.linalg.lstsq(A_x, x_coords.unsqueeze(1)).solution
                speed_x = coefficients_x[0]
            except:
                speed_x = torch.tensor(0.0, device=self.device)

            # Для Y координаты
            y_coords = recent_positions[:, 1]
            A_y = torch.stack([t_relative, torch.ones_like(t_relative)], dim=1)
            try:
                coefficients_y = torch.linalg.lstsq(A_y, y_coords.unsqueeze(1)).solution
                speed_y = coefficients_y[0]
            except:
                speed_y = torch.tensor(0.0, device=self.device)

            current_speed = torch.sqrt(speed_x ** 2 + speed_y ** 2)

            if torch.abs(current_speed) > 0.1:
                track_data['movement_angle'] = torch.atan2(speed_y, speed_x) * 180 / torch.pi

            # Расчет ускорения
            if track_data['speed_history_count'] > 0:
                if track_data['speed_history_count'] < 30:
                    track_data['speed_history'][track_data['speed_history_count']] = current_speed
                    track_data['speed_history_count'] += 1
                else:
                    track_data['speed_history'] = torch.roll(track_data['speed_history'], -1, 0)
                    track_data['speed_history'][-1] = current_speed

                if track_data['speed_history_count'] >= 10:
                    speed_history = track_data['speed_history'][:track_data['speed_history_count']]
                    t_accel = torch.arange(track_data['speed_history_count'],
                                           dtype=torch.float32, device=self.device)
                    A_accel = torch.stack([t_accel, torch.ones_like(t_accel)], dim=1)
                    try:
                        coefficients_accel = torch.linalg.lstsq(A_accel, speed_history.unsqueeze(1)).solution
                        acceleration = coefficients_accel[0]
                    except:
                        acceleration = torch.tensor(0.0, device=self.device)

                    smooth_factor = 0.9
                    track_data['world_acceleration'] = (
                            smooth_factor * track_data['world_acceleration'] +
                            (1 - smooth_factor) * acceleration
                    )
                    self.helping_data['sum_acc'] += track_data['world_acceleration']
                    self.helping_data["count_acc"] += 1
                    self.stats["avg_acc"] = self.helping_data['sum_acc'] / self.helping_data["count_acc"]
                    return current_speed, track_data['world_acceleration']

            return current_speed, torch.tensor(0.0, device=self.device)

        self.helping_data['sum_speed'] += track_data['world_speed']
        self.helping_data["count_speed"] += 1
        self.stats["avg_speed"] = self.helping_data['sum_speed'] / self.helping_data["count_speed"]
        return track_data['world_speed'], torch.tensor(0.0, device=self.device)

    def detect_lane_change_robust(self, track_data):
        """Детекция смены полосы"""
        if track_data['count'] < 30 or not track_data['is_calibrated']:
            return False

        valid_positions = track_data['positions'][:track_data['count']]

        if track_data['count'] >= 20:
            segment_size = track_data['count'] // 3
            segment1 = valid_positions[:segment_size]
            segment2 = valid_positions[segment_size:2 * segment_size]
            segment3 = valid_positions[2 * segment_size:]

            if len(segment1) >= 5 and len(segment2) >= 5 and len(segment3) >= 5:
                angle1 = self._calculate_movement_angle(segment1)
                angle2 = self._calculate_movement_angle(segment2)
                angle3 = self._calculate_movement_angle(segment3)

                angle_change_12 = torch.abs(angle1 - angle2)
                angle_change_23 = torch.abs(angle2 - angle3)

                significant_change = (angle_change_12 > self.LANE_CHANGE_ANGLE_THRESHOLD and
                                      angle_change_23 > self.LANE_CHANGE_ANGLE_THRESHOLD and
                                      torch.sign(angle1 - angle2) == torch.sign(angle2 - angle3))

                if significant_change:
                    self.stats['lane_count'] += 1
                return significant_change

        return False

    def _calculate_movement_angle(self, positions):
        """Вычисление угла движения"""
        if len(positions) < 2:
            return torch.tensor(0.0, device=self.device)

        t = torch.arange(len(positions), dtype=torch.float32, device=self.device)
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]

        A_x = torch.stack([t, torch.ones_like(t)], dim=1)
        try:
            coeff_x = torch.linalg.lstsq(A_x, x_coords.unsqueeze(1)).solution
            dx = coeff_x[0]
        except:
            dx = torch.tensor(0.0, device=self.device)

        A_y = torch.stack([t, torch.ones_like(t)], dim=1)
        try:
            coeff_y = torch.linalg.lstsq(A_y, y_coords.unsqueeze(1)).solution
            dy = coeff_y[0]
        except:
            dy = torch.tensor(0.0, device=self.device)

        if torch.abs(dx) > 0.001:
            angle = torch.atan2(dy, dx) * 180 / torch.pi
        else:
            angle = torch.tensor(90.0 if dy > 0 else -90.0, device=self.device)

        self.helping_data['sum_angle'] += angle
        self.helping_data["count_angle"] += 1
        self.stats["avg_angle"] = self.helping_data['sum_angle'] / self.helping_data["count_angle"]

        return angle

    def detect_aggressive_behavior(self, track_data, track_id, current_frame):
        """Детекция агрессивного поведения"""
        if track_data['count'] < self.MIN_TRACK_LENGTH or not track_data['is_calibrated']:
            return False, []

        speed, acceleration = self.calculate_world_metrics(track_data)
        behaviors = []
        speed_kmh = speed.item() * 3.6 if speed.numel() == 1 else 0.0
        acceleration_value = acceleration.item() if acceleration.numel() == 1 else 0.0

        # Фильтр нереалистичных скоростей
        if speed_kmh > 250.0 or speed_kmh < 1.0:
            return False, []

        # Проверка ускорения/торможения
        acceleration_significant = abs(acceleration_value) > 1.5

        if acceleration_significant:
            # Торможение
            if acceleration_value < self.DECELERATION_THRESHOLD.item():
                if speed_kmh > 50.0:
                    if track_data['aggressive_count'] == 0 or \
                            (current_frame - track_data['last_aggressive_frame']) > 60:
                        behaviors.append("HARD_BRAKING")
                        self.stats["br_count"] += 1

            # Ускорение
            elif acceleration_value > self.ACCELERATION_THRESHOLD.item():
                if speed_kmh > 30.0 and track_data['aggressive_count'] == 0:
                    behaviors.append("HARD_ACCELERATION")
                    self.stats["acc_count"] += 1

        # Смена полосы
        if self.detect_lane_change_robust(track_data):
            if current_frame - track_data['last_aggressive_frame'] > 90:
                behaviors.append("AGGRESSIVE_LANE_CHANGE")
                self.stats["lane_count"] += 1

        is_aggressive = len(behaviors) > 0

        if is_aggressive and (current_frame - track_data['last_aggressive_frame'] > 45):
            track_data['aggressive_count'] += 1
            track_data['last_aggressive_frame'] = current_frame
            self.stats['aggressive_count'] += 1

        return is_aggressive, behaviors

    def detect_aggressive_behavior_robust(self, track_data, current_frame):
        """Надёжная детекция агрессивного поведения"""
        if track_data['count'] < self.MIN_TRACK_LENGTH or not track_data['is_calibrated']:
            return False, []

        speed, acceleration = self.calculate_world_metrics(track_data)
        behaviors = []  # Переводим в км/ч для проверки
        speed_kmh = speed.item() * 3.6 if speed.numel() == 1 else 0.0
        acceleration_value = acceleration.item() if acceleration.numel() == 1 else 0.0

        # ФИЛЬТР НЕРЕАЛИСТИЧНЫХ СКОРОСТЕЙ
        if speed_kmh > 250.0 or speed_kmh < 1.0:
            return False, []

        # КОНСЕРВАТИВНЫЕ ПРОВЕРКИ

        # 1. Ускорение/торможение - требуем значительных изменений
        acceleration_significant = abs(acceleration_value) > 1.5

        if acceleration_significant:
            # Торможение - строгие условия
            if acceleration_value < self.DECELERATION_THRESHOLD.item():
                # Дополнительные проверки:
                # - Высокая исходная скорость
                # - Устойчивое торможение
                if speed_kmh > 50.0:  # > 50 км/ч
                    if track_data['aggressive_count'] == 0 or \
                            (current_frame - track_data['last_aggressive_frame']) > 60:
                        behaviors.append("REZKOE_TORMOZHENIE")
                        self.stats["br_count"] += 1

            # Ускорение - строгие условия
            elif acceleration_value > self.ACCELERATION_THRESHOLD.item():
                if speed_kmh > 30.0 and track_data['aggressive_count'] == 0:
                    behaviors.append("RESKOE_USKORENIE")
                    self.stats["acc_count"] += 1

        # 2. Смена полосы - надежная детекция
        if self.detect_lane_change_robust(track_data):
            # Только если не было недавних агрессивных событий
            if current_frame - track_data['last_aggressive_frame'] > 90:
                behaviors.append("REZKOE_PERESTROENIE")
                self.stats["lane_count"] += 1

        is_aggressive = len(behaviors) > 0

        # Консервативное обновление счетчиков
        if is_aggressive and (current_frame - track_data['last_aggressive_frame'] > 45):
            track_data['aggressive_count'] += 1
            track_data['last_aggressive_frame'] = current_frame
            self.stats['aggressive_count'] += 1

        return is_aggressive, behaviors
    def predict_aggressive_behavior(self, current_frame, frame_count):
        results = self.model.track(
            current_frame, persist=True, tracker="bytetrack.yaml",
            classes=[2, 3, 5, 7], conf=0.25, verbose=False, device=self.device
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.float().cpu().tolist()

            for box, track_id, conf in zip(boxes, track_ids, confidences):
                x, y, w, h = box
                center = torch.tensor([float(x), float(y)], device=self.device)

                self.update_track_history(track_id, center, torch.tensor(1, device=self.device))

                track_data = self.track_history[track_id]
                is_aggressive, behaviors = self.detect_aggressive_behavior_robust(
                    track_data, frame_count
                )

                return self.stats
