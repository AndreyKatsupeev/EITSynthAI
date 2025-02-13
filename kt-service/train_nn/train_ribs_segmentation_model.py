"""Скрипт для обучения нейросети для сегментации рёбер на фронтальных срезах"""

from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.train(data='data_ribs_segmentation_model.yaml', epochs=50, imgsz=320, device='cuda', batch=4)
results = model.train(data='data_ribs_segmentation_model.yaml', epochs=100, imgsz=320, device='cuda', batch=4)
results = model.train(data='data_ribs_segmentation_model.yaml', epochs=50, imgsz=640, device='cuda', batch=4)
results = model.train(data='data_ribs_segmentation_model.yaml', epochs=100, imgsz=640, device='cuda', batch=4)
results = model.train(data='data_ribs_segmentation_model.yaml', epochs=100, imgsz=448, device='cuda', batch=4)
results = model.train(data='data_ribs_segmentation_model.yaml', epochs=100, imgsz=448, device='cuda', batch=4)

