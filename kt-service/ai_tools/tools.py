import abc
import cv2
import numpy

from fsi_logger import write_log



class DICOMSeriesToMask():
    pass

class DICOMToMask():


class ImageToMask():
    pass


class NIIToMask():
    pass

class SearchSlice(abc.ABC):
    def __init__(self, model_path=None):
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = config.model_path
        self.model = self.__load_model(self.model_path)

    def __load_model(self, model_path):
        return model.load(model_path)

    def __get_slice(self, images):
        return slise

    def __ribs_predict(self, front_slice):
        pass


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
