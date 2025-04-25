import cv2
import logging
import zipfile
from io import BytesIO

from fastapi import FastAPI, Response
from .femm_generator import create_mesh

# Настройка логирования

from pydantic import BaseModel
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

class MeshData(BaseModel):
    params: List[float]  # Первые два числа (0.682, 0.682)
    polygons: List[str]   # Остальные строки с координатами

@app.post("/createMesh")
async def create_mesh_from_json(data: MeshData):
    """"""
    try:
        # Разбираем входные данные
        pixel_spacing = data.params[:2]
        polygons = data.polygons
        image_np = create_mesh(pixel_spacing, polygons)
        _, img_bytes = cv2.imencode(".png", image_np)
        img_bytes = img_bytes.tobytes()

        # Возвращаем бинарные данные
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        return {"status": "error", "message": str(e)}

    return answer
