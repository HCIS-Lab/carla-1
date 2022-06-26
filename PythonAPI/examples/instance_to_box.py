import cv2
from torchvision.io import read_image
import torch
import os 
import numpy as np
import time
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.ops.boxes import masks_to_boxes
from torchvision.io import read_image


def instance_to_box(mask,class_filter,threshold=50):
    """
        Args:
            mask: instance image
            class_filter: List[int] int: CARLA Semantic segmentation tags (classes that need bounding boxes)
        return:
            boxes: class_id, x1, y1, x2, y2
    """
    # mask = torch.tensor(instance).type(torch.int)
    mask_2 = torch.zeros(2,720,1280)
    mask_2[0] = mask[0]
    mask_2[1] = mask[1]+mask[2]*256
    condition = mask_2[0]==-1
    for class_id in class_filter:
        condition += mask_2[0]==class_id
    obj_ids = torch.unique(mask_2[1,condition])
    masks = mask_2[1] == obj_ids[:, None, None]
    area_condition = masks.long().sum((1,2))>=threshold
    masks = masks[area_condition]
    obj_ids = obj_ids[area_condition]
    boxes = masks_to_boxes(masks)
    class_ins = []
    for m in masks:
        unique, inverse = torch.unique(m.view(-1), sorted=True, return_inverse=True)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
        class_ins.append(mask_2[0].view(-1)[perm[1]])
    boxes = torch.cat((torch.stack(class_ins).unsqueeze(1),boxes),1)
    return boxes

def read_and_draw(scenario_path,write_video=True):
    class_dict = {4:'Pedestrian',10:'Vehicle',20:'Dynamic'}
    bboxes_file = sorted(os.listdir(os.path.join(scenario_path,'bbox/front')))[:-1]
    rgb_file = sorted(os.listdir(os.path.join(scenario_path,'rgb/front')))
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(os.path.join(scenario_path,'demo.mp4'), fourcc, 20.0, (1280,  720))
    for b_file,r_file in zip(bboxes_file,rgb_file):
        img = cv2.imread(os.path.join(scenario_path,'rgb/front',r_file))
        boxes = torch.load(os.path.join(scenario_path,'bbox/front',b_file))
        for box in boxes:
            id,x1,y1,x2,y2 = box.type(torch.int).numpy()
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
            cv2.putText(img,class_dict[id],(x1,y1),0,0.3,(0,255,0))
        out.write(img)
        cv2.imshow('result',img)
        time.sleep(0.05)
        c = cv2.waitKey(50)
        if c == ord('q') and c == 27:
            break
    out.release()
    cv2.destroyAllWindows()

def produce_boxes():
    root_path = 'data_collection'
    scenario_key = ['collision','interactive','non-interactive','obstacle']
    curr_path = root_path
    path = [curr_path]
    for scenario_type in os.listdir(curr_path):
        if scenario_type in scenario_key:
            curr_path = path[-1]
            curr_path = os.path.join(curr_path,scenario_type)
            path.append(curr_path)
            for scenario_id in os.listdir(curr_path):
                print(scenario_id)
                curr_path = path[-1]
                curr_path = os.path.join(curr_path,scenario_id,'variant_scenario')
                path.append(curr_path)
                for variant_name in os.listdir(curr_path):
                    print('\t',variant_name)
                    start_time = time.time()
                    curr_path = path[-1]
                    curr_path = os.path.join(curr_path,variant_name)
                    if not os.path.isdir(os.path.join(curr_path,'bbox')):
                        os.mkdir(os.path.join(curr_path,'bbox'))
                        os.mkdir(os.path.join(curr_path,'bbox','front'))
                    try:
                        instances = sorted(os.listdir(os.path.join(curr_path,'instance_segmentation/ins_front')))
                        for instance in instances:
                            frame_id = instance[:-4]
                            raw_instance = read_image(os.path.join(curr_path,'instance_segmentation/ins_front',instance))[:3].type(torch.int)
                            boxes = instance_to_box(raw_instance,[4,10,20])
                            # todo: add carla actor_id
                            torch.save(boxes,os.path.join(curr_path,'bbox/front/%s.pt' % (frame_id)))
                        print('time taken: {}'.format(time.time()-start_time))
                    except:
                        continue
                path.pop()
            path.pop()

if __name__ == '__main__':
    scenario_path = 'data_collection/obstacle/1_t3-2_0_f_sl/variant_scenario/WetNight_mid_'
    read_and_draw(scenario_path)
    # produce_boxes()
    
