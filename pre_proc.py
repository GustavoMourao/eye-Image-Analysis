from Interpreter import Interpreter
import cv2
from Processor import Processor
import matplotlib.pyplot as plt


def crop_image(images_info):
    """
    Crop image
    
    Args:
    ---------
        images_info: images path name
    
    Return:
    ---------
        save croped image
    """
    for namefile in images_info.filepaths:    
        img = cv2.imread(namefile)
        crop_img = img[0:1424, 0:1072]
        cv2.imwrite(namefile, crop_img)


def equalized_images(images_info):
    """
    Equalize images
    
    Args:
    ---------
        images_info: images path name
    
    Return:
    ---------
        save equalized image
    """
    proc = Processor(images_info[0])

    for namefile in images_info.filepaths:
        img = cv2.imread(namefile)
        img_equalized = proc.get_histogram_equalized(img)
        cv2.imwrite(
            namefile[:-4] + '_equalized.jpg',
            img_equalized
        )
        print(namefile[:-4] + '_equalized.jpg')


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

    images_info = inter.get_info_images('./Data/train')
    # crop_image(images_info)

    equalized_images(images_info)
