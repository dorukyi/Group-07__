import os
import shutil
import cv2
import numpy as np


'''This function returns TRUE if the mean luminosity ofan image is below a threshold'''

def isNight(img,threshold=0.1):
    L,a,b=cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2LAB))
    L=L/np.max(L)
    print(np.mean(L))
    return np.mean(L)<threshold



folder_dir= "C:/Users/Theo/Desktop/SE/Mixed Images"
folder_dir_day="C:/Users/Theo/Desktop/SE/Day Images"
folder_dir_night="C:/Users/Theo/Desktop/SE/Night Images"

'''This fo loop iterates through all images in a folder and sorts them in images taken by day or by night'''


for image in os.listdir(folder_dir):
    
    img1=cv2.imread(folder_dir+"/"+os.path.basename(image))
    if isNight(img1):
            shutil.move(folder_dir+"/"+os.path.basename(image),folder_dir_night)   
            print("Image moved to Night Images")                                           # <-- the image will be discarded in the real app
    else:
            shutil.move(folder_dir+"/"+os.path.basename(image),folder_dir_day)             # <-- the image will be fed to the model 
            print("Image moved to Day Images")