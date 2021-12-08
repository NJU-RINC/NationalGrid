import cv2
import numpy as np
# from scipy.misc import imread, imresize, imsave
from .func import *
from typing import Tuple, List
import numpy as np


def process(img: np.ndarray) -> np.ndarray:
    img = cv2.medianBlur(img, 7)
    cv2.normalize(img, img, alpha=0, beta=128, norm_type=cv2.NORM_MINMAX)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img= cv2.medianBlur(img, 5).astype(np.int16)

    return img

def api(image1: np.ndarray, image2: np.ndarray, expand_p: float) -> Tuple[np.ndarray, List[Box]]:
    image2_ori = image2

    # image1 = cv2.medianBlur(image1, 7)  # 使用7个卷积核进行中值化
    # image2 = cv2.medianBlur(image2, 7)

    # cv2.normalize(image1, image1, alpha=0, beta=128, norm_type=cv2.NORM_MINMAX)
    # cv2.normalize(image2, image2, alpha=0, beta=128, norm_type=cv2.NORM_MINMAX)
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # image1 = cv2.medianBlur(image1, 5).astype(np.int16)
    # image2 = cv2.medianBlur(image2, 5).astype(np.int16)

    image1 = process(image1)
    image2 = process(image2)

    diff_image = abs(image1 - image2)

    fig_size = float(image1.shape[0] * image1.shape[1])

    diff_image = (diff_image > 10) * 255

    change_map = diff_image.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    
    cleanChangeMap = change_map
    cleanChangeMap = cv2.erode(cleanChangeMap, kernel)  # 腐蚀操作
    cleanChangeMap = cv2.dilate(cleanChangeMap, kernel)  # 膨胀操作

    area_threshold = 0.001 * fig_size  # 1/1000就可以达成异常
    area_threshold2 = 0.1 * fig_size  # 认为不出现10%面积的异常，判定为光照因素

    box_list = get_box(cleanChangeMap, area_threshold, area_threshold2, image2_ori, 4, expand_p)#, multiple)

    # lens = len(box_list)
    #
    # for i in range(lens):
    #     plt.subplot(1, lens, i + 1)
    #     plt.imshow(box_list[i])
    #
    # # plt.savefig('test.jpg')
    # plt.show()
    # #return image_list

    return cleanChangeMap, box_list


