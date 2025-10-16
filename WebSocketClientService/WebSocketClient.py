import asyncio
import websockets
from websockets import ClientConnection
import logging
from typing import Dict, Optional
import uuid
import numpy
import json

from DetecorService.Detecor import AccurateGPUAggressiveDrivingDetector


class WebSocketClient:
    def __init__(self, uri: str):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.uri = uri
        self.websocket: ClientConnection = None
        self.is_connected = False
        self.client_id = str(uuid.uuid4())[:8]
        self.detector = AccurateGPUAggressiveDrivingDetector()

    async def connect(self):
        try:
            self.websocket = websockets.connect(self.uri)
            while True:
                await self.update_socket()
                await asyncio.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket server: {e}")
            return False

    async def update_socket(self) -> Optional[Dict]:
        if not self.is_connected or not self.websocket:
            return None

        data = await self.websocket.recv()
        data = json.loads(data)
        array = numpy.frombuffer(data["frame"], numpy.uint8)
        stats = self.detector.predict_aggressive_behavior(array, data["frame_id"])
        await self.websocket.send(json.dumps(stats))

    async def close(self):
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
