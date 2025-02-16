import cv2
import numpy

from fsi_logger import write_log

def image_decoder(list_images_binary):
    """
    Function for converting binary images to opencv format

    Args:
        list_images_binary: list of binary images from API

    Keyword arguments:
        None

    Raises:
        None

    Returns:
        list images opencv format

    """
    images_original_list = []
    for file in list_images_binary:
        arr = numpy.frombuffer(file.file.read(), dtype=numpy.uint8)
        img_np = cv2.imdecode(arr, 1)
        try:
            write_log("test_image_decoder", ['картинка с разрешением', img_np.shape])
        except:
            print("ошибка записи лога")
        images_original_list.append(img_np)
    return images_original_list