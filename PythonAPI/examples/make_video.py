import cv2
import numpy as np
import os 


import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument('--path', default='./', type=str)
    return parser

def index_min(list):
    return int(min(list).split(".")[0])

def index_max(list):
    return int(max(list).split(".")[0])

if __name__ == "__main__":
    
    #min_rbg_index = 
    
    
    parser = get_parser()
    args = parser.parse_args()
    path = args.path

    out = cv2.VideoWriter(path + "/label.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20,  (1280, 720) )
    
    rgb_left_index = os.listdir(path +"/rgb/left/")
    rgb_front_index = os.listdir(path +"/rgb/front/")
    rgb_right_index = os.listdir(path +"/rgb/right/")
    ss_lbc_index = os.listdir(path +"/semantic_segmentation/lbc_seg")
    
    min_list = [index_min(rgb_left_index), index_min(rgb_front_index), index_min(rgb_right_index), index_min(ss_lbc_index)]
    max_list = [index_max(rgb_left_index), index_max(rgb_front_index), index_max(rgb_right_index), index_max(ss_lbc_index)]
    start = max (min_list)
    end = min(max_list) + 1
    print(start, end)
    

    for index in range(start, end, 1 ):
        #print("%08d"%index)
        img_left = cv2.imread(path + "/rgb/left/%08d.png"%index)
        img_front = cv2.imread(path + "/rgb/front/%08d.png"%index)
        img_right = cv2.imread(path +"/rgb/right/%08d.png"%index)
        
        img_left = cv2.flip(img_left, 1)
        img_front = cv2.flip(img_front, 1)
        HomoMat = np.array([[-1.47259681e-01, -4.11121939e-02,  7.54217399e+02],
                            [-3.22292842e-01,  7.01744628e-01,  1.02357651e+02],
                            [-8.89971041e-04, -3.93292020e-05,  1.00000000e+00]])
        (hr, wr) = img_left.shape[:2]
        warp_img = cv2.warpPerspective(img_left, HomoMat, (wr*2, hr))
        warp_img[:,  : 1280] = img_front
        warp_img = cv2.flip(warp_img, 1)
        HomoMat = np.array([[-1.22652155e-01, -6.18352853e-02,  7.52281387e+02],
                            [-3.04984504e-01,  6.91283945e-01,  1.00266649e+02],
                            [-8.57714402e-04, -7.07633573e-05,  1.00000000e+00]])

        img_right = cv2.warpPerspective(img_right, HomoMat, (wr*2, hr))
        vis = np.concatenate((warp_img, img_right[:, 1280:2560]), axis=1)
        vis = cv2.resize(vis, (1280, 240), interpolation=cv2.INTER_AREA)
        
        img_top =  cv2.imread(path +"/semantic_segmentation/lbc_seg/%08d.png"%index)
        img_top = img_top[:256, :]
        img_top = cv2.resize(img_top, (940, 480), interpolation=cv2.INTER_AREA)
        
        
        img_top = cv2.copyMakeBorder(img_top, 0, 0, 170, 170, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # print(img_top.shape[:2])
        
        vis = np.concatenate((vis, img_top), axis=0)
        

        out.write(vis)
    out.release()
    

    
    