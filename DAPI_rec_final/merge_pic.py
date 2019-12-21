# -*- coding: utf-8 -*-
"""
Created on 03-12-2019
@author: Gou Yujie
"""
import cv2
import numpy as np
import os
import re
path1="F:\cuckoo_lab\image_project\pics\DAPI-Blue; TFEB-Red\positive"
path2="F:\cuckoo_lab\image_project\pics\DAPI-Blue; TFEB-Red\\negative"#\n是特殊字符
def walkFile(path):
    list_fold=[]
    list_pics_DAPI=[]
    list_pics_TFEB=[]
    for root,dirs,files in os.walk(path):
        for foldname in dirs:
            bpath=os.path.join(root,foldname) #所有path以下的文件夹名字
            if re.match(".*?[A-Z]$",bpath): #list_fold只留下绝对路径的倒数第二层文件夹
  #              print(bpath)
                list_fold.append(bpath)
  #  print(list_fold)
    for i in range(len(list_fold)): #建立一个列表来放两种染色图的文件名
        if re.search(".*-(\d-DAPI)", list_fold[i]) != None: #红色染料
            for file in os.listdir(list_fold[i]):
                list_pics_DAPI.append(list_fold[i]+'\\'+file)
        elif re.search(".*-(\d-TFEB)", list_fold[i]) != None:   #蓝色染料
            for file in os.listdir(list_fold[i]):
                list_pics_TFEB.append(list_fold[i]+'\\'+file)
        else:
            pass
#    print(list_pics_DAPI)
#    print(list_pics_TFEB)
    return list_pics_DAPI,list_pics_TFEB
list_pos_DAPI=walkFile(path1)[0]  #得到pos和neg分别红和蓝的图像路径列表
list_pos_TFEB=walkFile(path1)[1]
list_neg_DAPI=walkFile(path2)[0]
list_neg_TFEB=walkFile(path2)[1]
#print(list1)
#list2=walkFile(path2)
#print(len(list_neg_TFEB))
def getpic(list1,list2,type):  #图像merge，红色为主
    alpha = 0.25
    beta = 1 - alpha
    gamma = 0
    for i in range(len(list1)):
  #      for i in range(len(list2)):
        bottom_pic=cv2.imread(list1[i])
        top_pic=cv2.imread(list2[i])
        overlap_pic=cv2.addWeighted(bottom_pic,alpha,top_pic,beta,gamma)
        save_path="F:\cuckoo_lab\image_project\pics\merge\%s\\"%type
        img="%s_merge%d.jpg"%(type,i)
        cv2.imwrite(save_path+img,overlap_pic)
getpic(list_pos_DAPI,list_pos_TFEB,type='pos')
getpic(list_neg_DAPI,list_neg_TFEB,type='neg')




