import cv2
import asyncio
from ultralytics import YOLO
import logging
from typing import Dict, Any

from DetecorService.Detecor import AccurateGPUAggressiveDrivingDetector
from WebSocketClientService.WebSocketClient import WebSocketClient


class VideoProcessingService:
    def __init__(self, websocket_uri: str = "ws://127.0.0.1:5000/neural_ws"):
        self.detector = AccurateGPUAggressiveDrivingDetector()
        self.model = YOLO("yolo11n.pt")
        self.model.model.to(self.detector.device)
        self.frame_count = 0
        self.is_calibrated = False
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # WebSocket клиент для подключения к существующему серверу
        self.ws_client = WebSocketClient(websocket_uri)
        self.analysis_results = {}

    async def start_websocket_client(self):
        """Запуск WebSocket клиента"""
        return await self.ws_client.connect()

    async def process_img(self, image: bytes):
        """
        Обработка видео потока и отправка через WebSocket

        Args:
            video_source: путь к видеофайлу или номер камеры (0 для веб-камеры)
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video source: {video_source}")
            return

        self.logger.info(f"Started processing video stream from: {video_source}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.info("Video stream ended")
                    break

                # Кодирование кадра в JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpeg_data = buffer.tobytes()

                # Отправка через WebSocket
                result = self.detector.predict_aggressive_behavior()

                if result:
                    self.analysis_results = result
                    self.logger.info(f"Frame {self.frame_count}: {len(result.get('vehicles', {}))} vehicles")

                    # Вывод агрессивного поведения
                    for vehicle_id, vehicle_data in result.get('vehicles', {}).items():
                        if vehicle_data.get('is_aggressive'):
                            self.logger.warning(
                                f"Aggressive behavior detected - Vehicle {vehicle_id}: {vehicle_data['behaviors']}")

                self.frame_count += 1

                # Задержка для контроля FPS
                await asyncio.sleep(0.033)  # ~30 FPS

        except Exception as e:
            self.logger.error(f"Error processing video stream: {e}")
        finally:
            cap.release()
            await self.ws_client.close()

    def get_latest_results(self) -> Dict[str, Any]:
        """Получение последних результатов анализа"""
        return self.analysis_results