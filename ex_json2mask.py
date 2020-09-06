import numpy as np
import cv2
import json
import os
import os.path


file_list = os.listdir("../data/ex_jsons/ann")#Create list for file name
for file_name in file_list:
    with open(os.path.join("../data/ex_jsons/ann", file_name), "r") as src_file:
        data = src_file
        #Save File
        data_dict = json.load(data)
        #Partition
        data_objs=data_dict["objects"]
        d_w=data_dict['size']['width']
        d_h=data_dict['size']['height']
        for data in data_objs:
             obj_title = data['classTitle']
             if obj_title == "Freespace":
                obj_ext =data['points']
                points=data['points']['exterior']
                name=data['id']#Save id to use in file name
                img=np.zeros((d_h,d_w),dtype=np.uint8)#Create array with zero point
                cv2.fillPoly(img,np.array([points]),250)#Fill poly
                mask_name = str(name)+".png"
                mask_path =os.path.join("../data/ex_masks",mask_name)
                cv2.imwrite(mask_path, img)
                print(mask_name,'Work')#Tester
                    
                  
                   
                   

     
   