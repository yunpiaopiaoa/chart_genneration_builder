import cv2
from skimage.metrics import structural_similarity as ssim

def img_similarity(img1_path: str, img2_path: str) -> float:
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    similarity_index = ssim(img1, img2, data_range=img1.max() - img1.min())
    return similarity_index