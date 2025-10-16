import cv2
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import logging
from typing import Dict, Any, Optional

from VideoProcessingService.VideoProcessing import VideoProcessingService

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Aggressive Driving Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальный экземпляр сервиса
video_service: Optional[VideoProcessingService] = None


@app.on_event("startup")
async def startup_event():
    """Запуск сервиса при старте приложения"""
    global video_service
    video_service = VideoProcessingService()

    # Запуск WebSocket клиента в фоне
    success = await video_service.start_websocket_client()
    if success:
        logger.info("Video processing service started successfully")
    else:
        logger.error("Failed to start video processing service")


@app.on_event("shutdown")
async def shutdown_event():
    """Остановка сервиса при завершении приложения"""
    global video_service
    if video_service and video_service.ws_client:
        await video_service.ws_client.close()


@app.post("/api/start-processing")
async def start_processing(source: str = "0"):
    """Запуск обработки видео потока"""
    global video_service
    if not video_service:
        return {"status": "error", "message": "Service not initialized"}

    # Запуск обработки в фоновой задаче
    await asyncio.create_task(video_service.process_video_stream(source))
    return {"status": "started", "source": source}


@app.get("/api/results")
async def get_results():
    """Получение последних результатов анализа"""
    global video_service
    if not video_service:
        return {"status": "error", "message": "Service not initialized"}

    return video_service.get_latest_results()


@app.post("/api/calibration")
async def set_calibration(calibration_data: Dict[str, Any]):
    """Установка калибровочных параметров"""
    global video_service
    if not video_service:
        return {"status": "error", "message": "Service not initialized"}

    try:
        video_service.detector.set_calibration(calibration_data)
        video_service.is_calibrated = True
        return {"status": "calibration_set", "message": "Calibration data applied successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/health")
async def health_check():
    """Проверка статуса сервиса"""
    global video_service
    if not video_service:
        return {"status": "error", "message": "Service not initialized"}

    return {
        "status": "healthy",
        "device": str(video_service.detector.device),
        "calibration_status": video_service.is_calibrated,
        "websocket_connected": video_service.ws_client.is_connected,
        "total_frames_processed": video_service.frame_count
    }


@app.websocket("/ws/results")
async def websocket_results(websocket: WebSocket):
    """WebSocket для реального времени результатов"""
    await websocket.accept()
    logger.info("WebSocket results connection established")

    try:
        while True:
            global video_service
            if video_service:
                results = video_service.get_latest_results()
                await websocket.send_json(results)
            await asyncio.sleep(0.5)  # Отправка каждые 500ms

    except WebSocketDisconnect:
        logger.info("WebSocket results connection closed")
    except Exception as e:
        logger.error(f"WebSocket results error: {e}")


@app.get("/")
async def root():
    return {
        "message": "Aggressive Driving Detection Client API",
        "status": "running",
        "mode": "WebSocket Client"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)