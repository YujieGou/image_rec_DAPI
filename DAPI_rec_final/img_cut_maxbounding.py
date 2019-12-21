# -*- coding: utf-8 -*-
"""
Created on 03-12-2019
@author: Gou Yujie
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import  pandas as pd
from matplotlib import patches

def cut_maxbounding(imgpath,savepath):
    img = cv2.imread(imgpath)      #读入图像
    img2=np.copy(img)

    dst = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)#图像降噪
    Blur_img = cv2.medianBlur(dst, 5)  # 图像滤波
    img = cv2.cvtColor(Blur_img, cv2.COLOR_BGR2GRAY) #灰度化
    the =  cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,289, -3)#局部阈值形成二值图

    binary,contours,hierarchy=cv2.findContours(the,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    for i in range(len(contours)):#勾画边框
         #   approx1 = cv2.approxPolyDP(contours[i], 0.1, False)
        mom=cv2.moments(contours[i]) #记录边界位点
        if (mom['m00'] != 0 and contours[i].size>50):
            pt = (int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00'])) #计算中点
            cv2.circle(img, pt, 3, (255,255, 255), -1)  # 画白点
            x,y,w,h=cv2.boundingRect(contours[i])#取轮廓的最大外接矩形，太小的就不要了
            if (w > 30 and h > 30):
                crop_img = img2[y:y+h, x:x+w]
                crop_img=cv2.resize(crop_img,(56,56))      #切割的图统一大小
                seq = 'cut%s.jpg' % i
                cv2.imwrite(savepath + seq, crop_img)

imgpath='F:\cuckoo_lab\image_project\pics\merge\\testtest-0.25.jpg'
savepath = "F:\cuckoo_lab\image_project\pics\merge\cut_0.4\\"
cut_maxbounding(imgpath,savepath)
