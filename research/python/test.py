from Step1 import api as register
# from Step2 import api as detector
# from Step3 import api as recognizer
import numpy as np
import cv2 as cv

import time

a = f'data/yw_gkxfw_18.jpg'
b = f'data/yw_gkxfw_18_s2.jpg'


img1 = cv.imread(a)
img2 = cv.imread(b)

step1_start = time.time()

base, target = register(img1, img2)

step1_end= time.time()

cv.imwrite('base.jpg', base)
cv.imwrite('target.jpg', target)

# step2_start = time.time()
# cropped_list = detector(base, target)
# step2_end = time.time()

# step3_start = time.time()
# preds, *_ = recognizer(cropped_list)
# step3_end = time.time()

# print(preds)


# print(f'register time: {step1_end-step1_start}s; detect time: {step2_end-step2_start}; recognize time {step3_end-step3_start}')