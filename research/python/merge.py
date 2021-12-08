from __future__ import annotations
import numpy as np

ratio = 1.0

class Rect:
    def __init__(self, left, top, right, bottom) -> None:
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
    
    @staticmethod
    def setRatio(val):
        global ratio 
        ratio = val

    @property
    def x_ext(self):
        return (self.right - self.left) * (ratio - 1.0) // 2

    @property
    def y_ext(self):
        return (self.bottom - self.top) * (ratio - 1.0) // 2

    def __mul__(self, other: Rect):
        return self.left - self.x_ext < other.right + self.x_ext \
            and other.left - self.x_ext < self.right + self.x_ext \
            and self.top - self.y_ext < other.bottom + self.y_ext \
            and other.top - self.y_ext < self.bottom + self.y_ext

    def union(self, left, top, right, bottom):
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
    
    def __iadd__(self, other: Rect):
        self.union(other.left, other.top, other.right, other.bottom)
        return self

    def __repr__(self) -> str:
        return f'|left: {self.left}, top: {self.top}, right: {self.right}, bottom: {self.bottom}|'

            
rects = [Rect(0,0,2,2), Rect(1,1,3,3), Rect(2,2,4,4), Rect(5, 5, 7, 7), Rect(6,6,8,8), Rect(20,20, 22, 22)]

rects = np.array(rects)
# b = np.outer(a,a)
# np.fill_diagonal(b, False)
# print(b)

# c = np.any(b, axis=0).astype('bool')
# print(c)

# d = ~c
# print(d)

# a = a[list(c)]
# print(a)
# a = a[list(d)]
# print(a)

check = [False]*len(rects)
check = np.array(check).astype('bool')

Rect.setRatio(1.5)

for i in range(len(rects)-1):
    for j in range(i+1, len(rects)):
        if check[i]:
            break # try next i
        if rects[i]*rects[j]:
            rects[i] += rects[j]
            check[j] = True

print(check)

print(rects[~check])
