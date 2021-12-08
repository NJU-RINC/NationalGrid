from __future__ import annotations
import numpy as np
from skimage import measure
from typing import List, Tuple

class Box:
    def __init__(self, xmin: int, xmax: int, ymin: int, ymax: int):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

class Rect:
    ratio: float = 1.0
    w: int
    h: int
    def __init__(self, left, top, right, bottom) -> None:
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
    
    @staticmethod
    def set_ratio(val) -> None:
        Rect.ratio = val

    def set_backgroud_size(h, w) -> None:
        Rect.w = w
        Rect.h = h

    @property
    def x_ext(self) -> int:
        return int((self.right - self.left) * (Rect.ratio - 1.0) / 2)

    @property
    def y_ext(self) -> int:
        return int((self.bottom - self.top) * (Rect.ratio - 1.0) / 2)

    def expand(self, ratio = None) -> None:
        assert(Rect.w is not None and Rect.h is not None)
        if ratio is None:
            ratio = Rect.ratio
        x_ext = int((self.right - self.left) * (ratio - 1.0) / 2)
        y_ext = int((self.bottom - self.top) * (ratio - 1.0) / 2)
        self.left = max(self.left - x_ext, 0)
        self.top = max(self.top - y_ext, 0)
        self.right = min(self.right + x_ext, Rect.w)
        self.bottom = min(self.bottom + y_ext, Rect.h)

    def __mul__(self, other: Rect) -> bool:
        return self.left - self.x_ext < other.right + self.x_ext \
            and other.left - self.x_ext < self.right + self.x_ext \
            and self.top - self.y_ext < other.bottom + self.y_ext \
            and other.top - self.y_ext < self.bottom + self.y_ext

    def union(self, left, top, right, bottom) -> None:
        if ((left < right) and (top < bottom)):
            if ((self.left < self.right) and (self.top < self.bottom)):
                if self.left > left: self.left = left
                if self.top > top: self.top = top
                if self.right < right: self.right = right
                if self.bottom < bottom: self.bottom = bottom
            else:
                self.left = left
                self.top = top
                self.right = right
                self.bottom = bottom
    
    def __iadd__(self, other: Rect) -> Rect:
        self.union(other.left, other.top, other.right, other.bottom)
        return self

    def __repr__(self) -> str:
        return f'|left: {self.left}, top: {self.top}, right: {self.right}, bottom: {self.bottom}|'

def merge(image_list: List[Rect], backgroud_shape: Tuple[int, int], threshold: int = 1.0, factor: int = 1.0):
    assert(threshold >= 1.0)
    assert(1.0 <= factor <= threshold)
    Rect.set_backgroud_size(*backgroud_shape)
    Rect.set_ratio(threshold)
    rects = np.array(image_list)
    check = [False] * len(rects)
    check = np.array(check).astype('bool')

    for i in range(len(rects)-1):
        for j in range(i+1, len(rects)):
            if check[i]:
                break # try next i
            if rects[i]*rects[j]:
                rects[i] += rects[j]
                check[j] = True
        rects[i].expand(factor)

    return rects[~check]


def get_box(img, threshold_point,threshold_point2,image2,ab_num,expand_p):
    ################删去大小符合阈值的连通区域
    img_label, num = measure.label(img, return_num=True,connectivity=2)#输出二值图像中所有的连通域
    props = measure.regionprops(img_label)#输出连通域的属性，包括面积等

    res_area = []
    for i in range(1, len(props)):
        if props[i].area > threshold_point and props[i].area <threshold_point2:
            res_area.append(props[i])

    res_area.sort(key=lambda t: t.area,reverse=True)
    image_list=[]

    for i in range(min(len(res_area),ab_num)):
        col_min, row_min, col_max, row_max = res_area[i].bbox
        image_list.append(Rect(row_min, col_min, row_max, col_max))
    
    return merge(image_list, img.shape, 1.6, 1.2)