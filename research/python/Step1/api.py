from typing import Any, Tuple
import numpy as np

import cv2
import os
import numpy as np
from typing import List
import matplotlib.pyplot as plt

from Step1.scc import ssc

def api(base: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # do some manipulation
    im1Gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    MAX_FEATURES = 5000
    # 确定要查找的最大关键点数
    
    orb = cv2.ORB_create(MAX_FEATURES)
    
    # detect特征点、计算描述子
    # orb.detectAndCompute(im1Gray, None)
    # kp1, des1 = orb.detectAndCompute(im1Gray, None)
    # kp2, des2 = orb.detectAndCompute(im2Gray, None)
    kp1 = orb.detect(im1Gray)
    # img1 = cv2.drawKeypoints(im1Gray, kp1, outImage=None, color=(255, 0, 0))
    # cv2.imshow('key', img1)
    # cv2.waitKey(0)

    # kp1 = sorted(kp1, key=lambda x:x.response, reverse=True)
    # kp1 = ssc(kp1, 750, 0.1, im1Gray.shape[1], im1Gray.shape[0])

    # img2 = cv2.drawKeypoints(im1Gray, kp1, outImage=None, color=(255, 0, 0))
    # cv2.imshow('scc', img2)
    # cv2.waitKey(0)

    kp1, des1 = orb.compute(im1Gray, kp1)

    kp2 = orb.detect(im2Gray)
    # kp2 = sorted(kp2, key=lambda x:x.response, reverse=True)
    # kp2 = ssc(kp2, 750, 0.1, im2Gray.shape[1], im2Gray.shape[0])
    kp2, des2 = orb.compute(im2Gray, kp2)

    # 目前是暴力匹配
    # 交叉匹配则在后加True参数
    matcher = cv2.BFMatcher()#BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True) # 返回两个测试对象之间的汉明距离
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []

    for m,n in matches:
        if m.distance < 0.89 * n.distance:
            good.append([m])

    # 按相近程度排序
    # matches = sorted(matches, key=lambda x: x.distance)

    # keypoints_without_size = np.copy(target)
    # cv2.drawKeypoints(target, kp2, keypoints_without_size, color = (0, 255, 0))
    # result = cv2.drawMatches(base, kp1, target, kp2, matches, target, flags=2)
    # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # plt.show()
    
    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good])
    # 单应性矩阵计算
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RHO, 0.5)

    imOnes = np.ones_like(target, dtype=np.uint8)

    h, w = im2Gray.shape
    # print(im2Gray.shape)
    #im1Reg = cv2.warpPerspective(im2Gray, M, (w, h))

    baseReg = cv2.warpPerspective(base, M, (w, h), flags=cv2.INTER_LINEAR)

    imMask = cv2.warpPerspective(imOnes, M, (w, h), flags=cv2.INTER_LINEAR)
    #im2_mask = im1Gray * imMask

    targetMask = target * imMask
    
    
    return baseReg, targetMask


