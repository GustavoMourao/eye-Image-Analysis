from Interpreter import Interpreter
import cv2


if __name__ == "__main__":
    """
    Transform stereo image to mono stereo image.
    """
    BATCH_SIZE = 32
    IMAGE_SHAPE = (225, 225, 1)
    inter = Interpreter(
        BATCH_SIZE,
        IMAGE_SHAPE
    )
    images_info = inter.get_info_images('./Data/New-Database')

    for namefile in images_info.filepaths:    
        img = cv2.imread(namefile)
        crop_img = img[0:1424, 0:1072]
        cv2.imwrite(namefile, crop_img)
