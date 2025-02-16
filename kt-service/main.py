from ai_tools.tools import image_decoder
from ai_tools.fsi_logger import write_log

from fastapi import FastAPI, UploadFile, File, Request
from typing import List, Optional

app = FastAPI()


@app.post('/uploadDicom')
def dicom_handler(files: List[UploadFile] = File(...)):
    """

    Args:
        files:

    Returns:

    """

    list_image = image_decoder(files)
    number_out, max_proba_area, max_proba_image_index, error_message = numberRecognition(). \
        digits_detector(list_image, device, model_yolo_area, model_yolo_num)  # run neural networks
    write_log("out_digits_detector", [number_out, max_proba_area, max_proba_image_index, error_message])

    yolo_message = create_yolo_message(number_out, max_proba_area,
                                       max_proba_image_index, error_message)  # create yolo message response
    return yolo_message


@app.post('/uploadImage')
def image_handler(files: List[UploadFile] = File(...)):
    """

    Args:
        files:

    Returns:

    """
    pass