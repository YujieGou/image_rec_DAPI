# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:40:09 2019

@author: hancheng
"""

import cv2

def draw_contours(imgpath,t):
     img = cv2.imread(imgpath)
     Deno_img = cv2.fastNlMeansDenoisingColored(img, None, t, t, 7, 21)#图像降噪
     Blur_img = cv2.medianBlur(Deno_img,5)#图像滤波

     #灰度化
     gray_img = cv2.cvtColor(Blur_img, cv2.COLOR_BGR2GRAY)

     #局部阈值方法，所选参数比ADAPTIVE_THRESH_MEAN_C方法较优
     binary_img =  cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 367, 2)

     #全局阈值方法 ret, binary_img_1 = cv2.threshold(gray_img, 21, 255, cv2.THRESH_BINARY)

     #寻找轮廓
     _,contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     '''''
     delta = 0
     delete_list = []  # 新建待删除的轮廓序号列表
     c, row, col = hierarchy.shape
     for i in range(row):
          if hierarchy[0, i, 3] > 0:  # 有父轮廓或子轮廓
               delete_list.append(i)
     for i in range(len(delete_list)):
          # print("i= ", i)
          del contours[delete_list[i] - delta]
          delta = delta + 1
          '''''

     cv2.drawContours(img, contours, -1, (0, 225, 0), 1)
     #绘制轮廓
     for i in range(len(contours)):
         if (contours[i].size>120 and contours[i].size<1200):                              #点集大小在一定范围内才被判定为细胞，去除噪点
  #            approx1 = cv2.approxPolyDP(contours[i], 0.1, False)                           #对轮廓曲线进行平滑
  #            cv2.drawContours(img, [approx1], -1, (0, 255, 0), 2)
              M = cv2.moments(contours[i])                                                  # 计算第一条轮廓的各阶矩,字典形式
              center_x = int(M["m10"] / M["m00"])
              center_y = int(M["m01"] / M["m00"])
              cv2.circle(img, (center_x, center_y), 3, (0,255,0), -1)                              #绘制中心点
     return img
     #cv2.imshow("img", binary_img)
     #cv2.waitKey(0)
imgpath="F:/cuckoo_lab/image_project/pics/merge/testtest-0.4.jpg"
img=draw_contours(imgpath,t=25)
cv2.imwrite('F:\cuckoo_lab\image_project\pics\merge\cell_identified_0.4_2.jpg', img)