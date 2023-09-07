import os 
import cv2
import numpy as np


if __name__ =="__main__":
    pass

    file_list = os.listdir("./no_mask")
    for file in file_list:
        mask_video_list = []
        no_mask_video_list = []
        mask_video = cv2.VideoCapture(f"./mask/{file}")
        while(True):
            ret, frame = mask_video.read()
            if ret is False:
                break
            mask_video_list.append(frame)


        no_mask_video = cv2.VideoCapture(f"./no_mask/{file}")
        while(True):
            ret, frame = no_mask_video.read()
            if ret is False:
                break
            no_mask_video_list.append(frame)
            

        if not os.path.exists(f"./merge"):
            os.makedirs(f"./merge")

        
        max_len = max(len(mask_video_list), len(no_mask_video_list))

        out = cv2.VideoWriter(f'./merge/{file}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20,  (256, 512)) 

        for frame_num in range(max_len):
            # print(frame_num)
            if frame_num < len(mask_video_list):
                mask_img = mask_video_list[frame_num]
            else:
                mask_img = np.zeros((256, 256, 3), np.uint8)

            if frame_num < len(no_mask_video_list):
                no_mask_img = no_mask_video_list[frame_num]
            else:
                
                no_mask_img = np.zeros((256, 256, 3), np.uint8)

            concate_img = cv2.vconcat([mask_img, no_mask_img])

            cv2.putText(concate_img,"mask",(0,30),cv2.FONT_HERSHEY_COMPLEX, 1,(0, 255, 0), 3) 
            cv2.putText(concate_img,"no mask",(0,286),cv2.FONT_HERSHEY_COMPLEX, 1,(0, 255, 0), 3) 

            out.write(concate_img)
    
        out.release()
        



        