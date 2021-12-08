from typing import Tuple
import numpy as np

import cv2

def test(base):
    base: np.ndarray = np.frombuffer(base)

    return base.tobytes()

def api(base, target):
    base = bytes(base)
    target = bytes(target)

    base = np.frombuffer(base)
    target = np.frombuffer(target)

    base = np.reshape()
    # do some manipulation
    im1Gray = cv2.cvtColor(base, cv2.COLOR_RBG2GRAY)
    im2Gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    MAX_FEATURES = 5000
    # 确定要查找的最大关键点数
    
    orb = cv2.ORB_create(MAX_FEATURES)
    
    # detect特征点、计算描述子
    kp1 = orb.detect(im1Gray)
    kp1, des1 = orb.compute(im1Gray, kp1)
    kp2 = orb.detect(im2Gray)
    kp2, des2 = orb.compute(im2Gray, kp2)
    # 目前是暴力匹配
    # 交叉匹配则在后加True参数
    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING) # 返回两个测试对象之间的汉明距离
    matches = matcher.match(des1, des2)
    # 按相近程度排序
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    # 单应性矩阵计算
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 0.5)
    im_mask = np.ones_like(base, dtype=np.uint8)
    h, w = im2Gray.shape
    # print(im2Gray.shape)
    #im1Reg = cv2.warpPerspective(im2Gray, M, (w, h))

    baseReg: np.ndarray= cv2.warpPerspective(base, M, (w, h))
    imMask = cv2.warpPerspective(im_mask, M, (w, h))

    
    return baseReg.tobytes()
