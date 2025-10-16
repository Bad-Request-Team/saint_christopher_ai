import websockets
from websockets import ClientConnection
import logging
from typing import Dict, Optional
import uuid
import numpy
import json
from PIL import Image
import torch

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
        self.count_frames = 0

    async def connect(self):
        try:
            async with websockets.connect(self.uri, max_size=2**32) as websocket:
                self.logger.info("Connected")
                await self.update_socket(websocket)
        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket server: {e}")
            return False

    async def update_socket(self, websocket: ClientConnection) -> Optional[Dict]:
        data = await websocket.recv()
        self.logger.info(data)
        if data is not None:
            self.count_frames += 1
            image = Image.frombytes("RGB", (640, 640), data)
            image = torch.tensor(numpy.asarray(image), dtype=torch.float32).permute(2, 0, 1)
            image = image.unsqueeze(0)
            stats = self.detector.predict_aggressive_behavior(image, self.count_frames)
            await websocket.send(json.dumps(stats))

    async def close(self):
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
