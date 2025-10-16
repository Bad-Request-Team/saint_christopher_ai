import json
import websockets
import logging
from typing import Dict, Optional
import base64
import uuid


class WebSocketClient:
    def __init__(self, uri: str):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.uri = uri
        self.websocket = None
        self.is_connected = False
        self.client_id = str(uuid.uuid4())[:8]

    async def connect(self):
        """Подключение к WebSocket серверу"""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.is_connected = True
            self.logger.info(f"Connected to WebSocket server {self.uri}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket server: {e}")
            return False

    async def send_frame(self, frame_data: bytes) -> Optional[Dict]:
        """Отправка кадра и получение результата"""
        if not self.is_connected or not self.websocket:
            if not await self.connect():
                return None

        try:
            # Кодируем кадр в base64
            frame_b64 = base64.b64encode(frame_data).decode('utf-8')
            await self.websocket.send(frame_b64)

            # Получаем ответ
            response = await self.websocket.recv()
            result = json.loads(response)
            return result

        except Exception as e:
            self.logger.error(f"Error in WebSocket communication: {e}")
            self.is_connected = False
            return None

    async def close(self):
        """Закрытие соединения"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False