import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
root_path = "./NIMS_image/Total_checked/"
S_value=[]
S_label=[]
for file in os.listdir(root_path):
    path=root_path+file
    src = cv.imread(path,0) # 直接以灰度图方式读入
    try:
        img = src.copy()
        # 计算sobel卷积结果
        x = cv.Sobel(img,cv.CV_16S,1,0)
        y = cv.Sobel(img,cv.CV_16S,0,1)
        # 转换数据 并 台成
        # 格式转换函数
        Scale_absX = cv.convertScaleAbs(x)
        Scale_absY = cv.convertScaleAbs(y)
        result = cv.addWeighted(Scale_absX,0.5,Scale_absY,0.5,0) # 图像混合
        print(file[-5],np.mean(result))
        S_value.append(np.mean(result))
        S_label.append(file[-5])
    except:
        print('error')