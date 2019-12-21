# -*- coding: utf-8 -*-
"""
Created on 03-12-2019
@author: Gou Yujie
"""
from cell_identification import  draw_contours #从之前写的模块中调用
from img_cut_maxbounding import cut_maxbounding
import os
import cv2
path1="F:\cuckoo_lab\image_project\pics\merge\pos"
path2="F:\cuckoo_lab\image_project\pics\merge\\neg"
def cut_pics(path,type):
    for root, dirs, files in os.walk(path):
        i=0
        for f in files:
            n=os.path.join(root, f)
            print(n)
            img=cv2.imread(n)
            i+=1
            savepath = "F:\cuckoo_lab\image_project\pics\merge\%s_cut_%s\\" % (type, i)
            os.mkdir(savepath)
            cut_maxbounding(imgpath=n,savepath=savepath)
    #        name='img_%s_%d.png' % (type, i)
        #    imgpath="F:\cuckoo_lab\image_project\pics\merge\

      #      cv2.imwrite(apath+name,img)
cut_pics(path1,type='pos')
cut_pics(path2,type='neg')