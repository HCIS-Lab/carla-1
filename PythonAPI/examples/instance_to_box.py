import os 
import argparse
import time
import json
import csv
import cv2
from matplotlib.pyplot import acorr
from torchvision.io import read_image
import torch
from torchvision.ops.boxes import masks_to_boxes
from torchvision.io import read_image

def instance_to_box(mask,class_filter,actor_id_list,threshold=60):
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
    # def generate_boxes(class_id):
    #     condition = mask_2[0]==class_id
    #     obj_ids = torch.unique(mask_2[1,condition])
    #     filter_exist_actor = []
    #     for obj_id in obj_ids:
    #         obj_id = int(obj_id.numpy())
    #         if obj_id in actor_id_list:
    #             filter_exist_actor.append(True)
    #         else:
    #             filter_exist_actor.append(False)
    #     obj_ids = obj_ids[filter_exist_actor]
    #     masks = mask_2[1] == obj_ids[:, None, None]
    #     masks = masks*condition
    #     area_condition = masks.long().sum((1,2))>=threshold
    #     masks = masks[area_condition]
    #     obj_ids = obj_ids[area_condition].type(torch.int).numpy()
    #     boxes = masks_to_boxes(masks).type(torch.int16).numpy()
    #     out_list = []
    #     for id,box in zip(obj_ids,boxes):
    #         out_list.append({'actor_id':int(id),'class':int(class_id),'box':box.tolist()})
    #     return out_list
    
    h,w = mask.shape[1:]
    mask_2 = torch.zeros(2,h,w)
    mask_2[0] = mask[0]
    mask_2[1] = mask[1]+mask[2]*256

    # ped,vehicle
    condition = mask_2[0]== 4
    condition += mask_2[0]== 10
    obj_ids = torch.unique(mask_2[1,condition])
    filter_exist_actor = []
    for obj_id in obj_ids:
        obj_id = int(obj_id.numpy())
        if obj_id in actor_id_list[0]:
            filter_exist_actor.append(True)
        else:
            filter_exist_actor.append(False)
    obj_ids = obj_ids[filter_exist_actor]
    masks = mask_2[1] == obj_ids[:, None, None]
    masks = masks*condition
    area_condition = masks.long().sum((1,2))>=threshold
    masks = masks[area_condition]
    obj_ids = obj_ids[area_condition].type(torch.int).numpy()
    boxes = masks_to_boxes(masks).type(torch.int16).numpy()
    out_list = []
    for id,box in zip(obj_ids,boxes):
        out_list.append({'actor_id':int(id),'class':int(actor_id_list[0][id]),'box':box.tolist()})
    # obstacle
    condition += mask_2[0]== 20
    obj_ids = torch.unique(mask_2[1,condition])
    filter_exist_actor = []
    for obj_id in obj_ids:
        obj_id = int(obj_id.numpy())
        if obj_id in actor_id_list[1]:
            filter_exist_actor.append(True)
        else:
            filter_exist_actor.append(False)
    obj_ids = obj_ids[filter_exist_actor]
    masks = mask_2[1] == obj_ids[:, None, None]
    masks = masks*condition
    area_condition = masks.long().sum((1,2))>=threshold
    masks = masks[area_condition]
    obj_ids = obj_ids[area_condition].type(torch.int).numpy()
    boxes = masks_to_boxes(masks).type(torch.int16).numpy()
    for id,box in zip(obj_ids,boxes):
        out_list.append({'actor_id':int(id),'class':20,'box':box.tolist()})
    # out_list = []
    # for class_id in class_filter:
    #     out_list += generate_boxes(class_id)
    
    return out_list

def read_and_draw(scenario_path,write_video=True):
    # class_dict = {4:'Pedestrian',10:'Vehicle',20:'Dynamic',9:'Wrong!'}
    bboxes_file = sorted(os.listdir(os.path.join(scenario_path,'bbox/front')))
    rgb_file = sorted(os.listdir(os.path.join(scenario_path,'rgb/front')))
    counter = 0
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
        cv2.putText(img,str(counter),(10,50),0,2.0,(0,255,0))
        out.write(img)
        cv2.imshow('result',img)
        time.sleep(0.05)
        c = cv2.waitKey(50)
        if c == ord('q') and c == 27:
            break
        counter += 1 
    out.release()
    cv2.destroyAllWindows()

def generate_actor_ids(rows):
    out = {}
    out_2 = []
    for row in rows: 
        if int(row[1])==20:
            out_2.append(int(row[0])&0xffff)
        else:
            out[int(row[0])&0xffff] = int(row[1])
    return out,out_2

def produce_boxes():
    root_path = 'data_collection'
    # scenario_key = ['collision','interactive','non-interactive','obstacle']
    scenario_key = ['collision','interactive']
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
                    
                    with open(os.path.join(curr_path,'actor_list.csv'), newline='') as csvfile:
                        rows = list(csv.reader(csvfile))[1:]
                    actor_id_list = generate_actor_ids(rows)
                    try:
                        instances = sorted(os.listdir(os.path.join(curr_path,'instance_segmentation/ins_front')))
                        for instance in instances:
                            frame_id = instance[:-4]
                            raw_instance = read_image(os.path.join(curr_path,'instance_segmentation/ins_front',instance))[:3].type(torch.int)
                            data = instance_to_box(raw_instance,[4,10,20],actor_id_list)
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
    
