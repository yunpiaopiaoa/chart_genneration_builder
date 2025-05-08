from abc import abstractmethod


class BaseImgGenerator:
    def __init__(self):
        pass

    @abstractmethod
    def generate_img(self, code: str, save_path: str):
        pass
    def cleanup(self):
        pass
