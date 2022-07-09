import os 
import argparse
import time
import json
import csv
import cv2
from torchvision.io import read_image
import torch
from torchvision.ops.boxes import masks_to_boxes
from torchvision.io import read_image

def instance_to_box(mask,class_filter,threshold=60):
    """
        Args:
            mask: instance image
            class_filter: List[int] int: CARLA Semantic segmentation tags (classes that need bounding boxes)
        return:
            boxes: List[Dict], 
                key: 
                    actor_id: carla actor id & 0xffff, 
                    class: carla segmentation tag, 
                    box: bounding box(x1,y1,x2,y2)
    """
    # mask = torch.tensor(instance).type(torch.int)
    h,w = mask.shape[1:]
    mask_2 = torch.zeros(2,h,w)
    mask_2[0] = mask[0]
    mask_2[1] = mask[1]+mask[2]*256
    condition = mask_2[0]==-1
    for class_id in class_filter:
        condition += mask_2[0]==class_id
    obj_ids = torch.unique(mask_2[1,condition])
    masks = mask_2[1] == obj_ids[:, None, None]
    area_condition = masks.long().sum((1,2))>=threshold
    masks = masks[area_condition]
    obj_ids = obj_ids[area_condition].type(torch.int).numpy()
    boxes = masks_to_boxes(masks).type(torch.int16).numpy()
    mask_2 = mask_2.type(torch.int8).view(2,-1)
    class_ins = []
    for m in masks:
        unique, inverse = torch.unique(m.view(-1), sorted=True, return_inverse=True)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
        class_ins.append(mask_2[0][perm[1]])
    if len(class_ins)!=0:
        class_ins = torch.stack(class_ins).numpy()
    out_list = []
    for id,class_id,box in zip(obj_ids,class_ins,boxes):
        if int(class_id) ==9:
            class_id = 20
        out_list.append({'actor_id':int(id),'class':int(class_id),'box':box.tolist()})
    # boxes = torch.cat((torch.stack(class_ins).unsqueeze(1),boxes),1)
    return out_list

def read_and_draw(scenario_path,write_video=True):
    # class_dict = {4:'Pedestrian',10:'Vehicle',20:'Dynamic',9:'Wrong!'}
    bboxes_file = sorted(os.listdir(os.path.join(scenario_path,'bbox/front')))
    rgb_file = sorted(os.listdir(os.path.join(scenario_path,'rgb/front')))

    with open(os.path.join(scenario_path,'actor_list.csv'), newline='') as csvfile:
        rows = list(csv.reader(csvfile))[1:]
        for row in rows:
            print(row[0],int(row[0])&0xffff,row[1])
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(os.path.join(scenario_path,'demo.mp4'), fourcc, 20.0, (1280,  720))
    for b_file,r_file in zip(bboxes_file,rgb_file):
        img = cv2.imread(os.path.join(scenario_path,'rgb/front',r_file))
        # boxes = torch.load(os.path.join(scenario_path,'bbox/front',b_file))
        with open(os.path.join(scenario_path,'bbox/front',b_file)) as json_file:
            datas = json.load(json_file)
        for data in datas:
            id = data['actor_id']
            x1,y1,x2,y2 = data['box']
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
            cv2.putText(img,str(id),(x1,y1),0,0.3,(0,255,0))
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
                            data = instance_to_box(raw_instance,[4,10,20])
                            # raise
                            # torch.save(boxes,os.path.join(curr_path,'bbox/front/%s.pt' % (frame_id)))
                            with open(os.path.join(curr_path,'bbox/front/%s.json' % (frame_id)), 'w') as f:
                                json.dump(data, f)
                        print('time taken: {}'.format(time.time()-start_time))
                    except Exception as e:
                        print(e)
                        continue
                path.pop()
            path.pop()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--mode',
        default='box',
        required=True,
        help='mode, produce bounding boxes or demo')
    argparser.add_argument(
        '--path',
        default=None,
        required=False,
        help='scenario path')

    args = argparser.parse_args()
    if args.mode == 'box':
        produce_boxes()
    elif args.mode == 'demo':
        assert args.path is not None
        read_and_draw(args.path)
    
