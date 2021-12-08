import numpy as np
import cv2 as cv
from Step1 import api as register
from Step2 import api as detector
from typing import List, Tuple
import matplotlib.pyplot as plt


def imgs_show(imgs: List[np.ndarray]):
    for idx, img in enumerate(imgs):
        plt.subplot(1, len(imgs), idx+1)
        plt.axis("off")
        plt.imshow(img)


# read video file
cap = cv.VideoCapture('data/test.mp4')

# get info of total number of frames
cnt = cap.get(cv.CAP_PROP_FRAME_COUNT)
print(cnt)

# set start frame id which starts from 0.0
cnt = cap.set(cv.CAP_PROP_POS_FRAMES, 10.0)
ret, frame = cap.read() # read() will always get next frame, so var 'frame' here get 11.0th frame

# img1 as base img
img1 = frame
#img1 = frame[60:900, 44:500:, :] # avoid camera broder distortion

# again set the start frame id so that defect directly occurs
cnt = cap.set(cv.CAP_PROP_POS_FRAMES, 150.0)

h, w, _ = frame.shape

#out = cv.VideoWriter('out.mp4', cv.VideoWriter_fourcc(*'mp4v'), 10, (w*3, h))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # img2 as target img with defect
    img2 = frame #[60:900, 44:500:, :]
    base, target = register(img1, img2) # @base: registed base img; @target: masked target img with defect

    # blend registed base img and unmasked target img in 3rd channel
    img3 = img2
    p = 0.5
    img3[:,:,2] = base[:,:,2] * p + (1-p)* img3[:,:,2]

    # @diff: diff img right before boxes finding @boxes: rects that cover the defect region
    diff, boxes = detector(base, target, expand_p=1.2)

    for box in boxes:
        img2 = cv.rectangle(img2, (box.left, box.top), (box.right, box.bottom), (255,255,0), thickness=2)
        diff = cv.rectangle(diff, (box.left, box.top), (box.right, box.bottom), (255,0,0), thickness=0)
    
    diff = cv.cvtColor(diff, cv.COLOR_GRAY2BGR)
    total = np.hstack([base, img3, diff])
    # cv.imshow('base', base)
    # cv.imshow('diff', diff)
    # cv.imshow('frame', img3)
    # out.write(total)
    cv.imshow('total', total)

    
    

    if cv.waitKey(1) == ord('q'):
        break
plt.show()
    
cap.release()
#out.release()
cv.destroyAllWindows()
