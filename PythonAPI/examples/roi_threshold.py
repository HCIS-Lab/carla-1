import numpy as np
import json
import os
import matplotlib.pyplot as plt


def get_gt(data_type, root, scenario_weather):

    temp = scenario_weather.split('_')
    scenario_id = '_'.join(temp[:-3])
    weather = '_'.join(temp[-3:-1])+'_'

    basic_scene = os.path.join(root, scenario_id)
    var_scene = os.path.join(basic_scene, 'variant_scenario', weather)

    dyn_desc = open(os.path.join(var_scene, 'dynamic_description.json'))
    data = json.load(dyn_desc)
    dyn_desc.close()

    gt_cause_id = -1
    frame_id = -1

    if data_type == 'interactive':
        for key in data.keys():
            if key.isdigit():
                gt_cause_id = data[key]
                break

        behavior_change_path = os.path.join(
            basic_scene, 'behavior_annotation.txt')
        behavior_change_file = open(behavior_change_path, 'r')
        temp = behavior_change_file.readlines()
        frame_id = (int(temp[1].strip()) + int(temp[0].strip()))//2

    elif data_type == 'obstacle':

        obstacle_type = int(scenario_id.split('_')[2])

        behavior_change_path = os.path.join(
            var_scene, 'behavior_annotation.txt')
        behavior_change_file = open(behavior_change_path, 'r')
        temp = behavior_change_file.readlines()

        start_frame = int(temp[0].strip())
        end_frame = int(temp[1].strip())

        frame_id = ((end_frame-start_frame)*2//3) + start_frame

        if obstacle_type == 3:
            gt_cause_id = int(temp[2].strip())

        else:
            bbox_path = os.path.join(
                var_scene, 'bbox', 'front', f'{frame_id:08d}.json')
            bbox_json = open(bbox_path, 'r')
            bbox_data = json.load(bbox_json)
            bbox_json.close()

            gt_cause_list = []
            for actor in bbox_data:
                if actor['class'] == 20:
                    gt_cause_list.append(actor['actor_id'])

            gt_cause_id = gt_cause_list

    elif data_type == 'collision':
        for key in data.keys():
            if key.isdigit():
                gt_cause_id = data[key]
                break

        bbox_folder = os.path.join(var_scene, 'bbox', 'front')
        frames = os.listdir(bbox_folder)
        frames.sort()

        for frame in frames[::-1]:
            bbox_path = os.path.join(bbox_folder, frame)
            bbox_json = open(bbox_path, 'r')
            bbox_data = json.load(bbox_json)
            bbox_json.close()

            for actor in bbox_data:
                if actor['actor_id'] == gt_cause_id:
                    frame_id = int(frame.split('.')[0])-5
                    break

            if frame_id != -1:
                break

    elif data_type == 'non-interactive':

        rgb_path = os.path.join(var_scene, 'bbox', 'front')
        frames = os.listdir(rgb_path)
        frames.sort()
        start_frame = int(frames[0].split('.')[0])
        end_frame = int(frames[-1].split('.')[0])

        frame_id = ((end_frame-start_frame)*2//3) + start_frame

    if not isinstance(gt_cause_id, list):
        gt_cause_id = [gt_cause_id]

    return gt_cause_id, frame_id


def testing(data_type, root, RA, threshold, T):

    TP, FN, FP, TN = 0, 0, 0, 0

    for scenario_weather in RA.keys():

        # print(scenario_weather)

        # get groundtruth cause object id
        gt_cause_id, frame_id = get_gt(data_type, root, scenario_weather)

        if frame_id == -1:
            continue

        risk_object = True
        threshold_in_a_row = True
        risk_id = -1

        for frame in range(frame_id, frame_id-T, -1):
            if RA[scenario_weather][str(frame)]["scenario_go"] > threshold:
                threshold_in_a_row = False

            highest = (-999, 0)   # (id, score)

            for obj_id in RA[scenario_weather][str(frame)]:
                if obj_id.isdigit():
                    risk_score = RA[scenario_weather][str(frame)][obj_id]
                    if risk_score > highest[1]:
                        highest = (int(obj_id), risk_score)

            if risk_id == -1:
                risk_id = highest[0]

            elif risk_id != highest[0]:
                risk_object = False

        if threshold_in_a_row:
            if data_type == 'non-interactive':
                if risk_object:
                    FP += 1
                else:
                    TN += 1
            else:
                if risk_object:
                    if risk_id in gt_cause_id:
                        TP += 1
                    else:
                        FP += 1
                else:
                    FN += 1
        else:
            if data_type == 'non-interactive':
                TN += 1
            else:
                FN += 1

    return np.array([TP, FN, FP, TN])


def table(roi_threshold, sliding_window, record, save):

    rowLabel = []
    colLabel = []

    for t in sliding_window:
        rowLabel.append(f"T={t}")
    for threshold in roi_threshold:
        colLabel.append(f"Threshold = {threshold:.2f}")

    fig = plt.figure(figsize=(9, len(sliding_window)))
    plt.table(cellText=record.T.tolist(), colLabels=colLabel,
              rowLabels=rowLabel, loc='center', cellLoc='center')

    plt.axis('off')
    plt.grid('off')
    fig.set_size_inches(18.5, 10.5)

    if save:
        plt.savefig(f'table.png', dpi=200)
    plt.show()


def plot(s_go_threshold, sliding_window, record, save, metric='metric'):
    for result, t in zip(record.T, sliding_window):
        plt.plot(s_go_threshold, result, label=f'T={t}')

    plt.xlabel('Confidence go (s_go) threshold')
    plt.xticks(s_go_threshold)
    plt.ylabel(f'ROI {metric}')
    # plt.yticks(np.arange(0.4, 1, 0.1))

    plt.title(
        f'ROI by different sliding window size and s_go threshold')
    plt.legend(loc='upper right', prop={'size': 8})
    
    if save:
        plt.savefig(f"ROI_{metric}.png", dpi=200)
    plt.show()


def read_data(data_type):

    data_root = f'/media/waywaybao_cs10/Disk_2/Retrieve_tool/data_collection/{data_type}'
    json_path = f'RA/RA_source/RA_{data_type}_timesteps=5.json'

    RA_file = open(json_path)
    RA = json.load(RA_file)
    RA_file.close()

    return data_root, RA


def compute_f1(confusion_matrix):

    recall = confusion_matrix[0] / (confusion_matrix[0]+confusion_matrix[1])
    precision = confusion_matrix[0] / (confusion_matrix[0]+confusion_matrix[2])
    f1_score = 2*precision*recall / (precision+recall)

    return recall, precision, f1_score


def main():

    data_type = ['interactive', 'obstacle', 'collision', 'non-interactive']
    save_result = True
    show_process = True

    s_go_threshold = list(np.arange(0.2, 0.65, 0.05))
    sliding_window = list(np.arange(1, 6))
    record = np.zeros((len(s_go_threshold), len(sliding_window), 3))

    for i, threshold in enumerate(s_go_threshold):
        for j, T in enumerate(sliding_window):
            confusion_matrix = np.zeros((4))

            for _type in data_type:
                data_root, RA = read_data(_type)

                # confusion_matrix: 1*4 ndarray, [TP, FN, FP, TN]
                confusion_matrix += testing(_type, data_root, RA, threshold, T)

            recall, precision, f1_score = compute_f1(confusion_matrix)
            record[i, j] = recall, precision, f1_score

            if show_process:
                print(f"Threshold: {threshold:.4f}, Sliding window: {T}")
                print(
                    f"Recall: {recall:.4f}  Precision{precision:.4f}  F1-Score{f1_score:.4f}")
                print("=======================================================")

    record = np.around(record, decimals=4)

    print("Recall:")
    print(record[:, :, 0])
    print()
    print("Precision:")
    print(record[:, :, 1])
    print()
    print("F1-Score:")
    print(record[:, :, 2])

    plot(s_go_threshold, sliding_window, record[:, :, 0], save_result, 'recall')
    plot(s_go_threshold, sliding_window, record[:, :, 1], save_result, 'Precision')
    plot(s_go_threshold, sliding_window, record[:, :, 2], save_result, 'F1-Score')
    # table(roi_threshold, sliding_window, record[:, :, 2], save_result)


if __name__ == '__main__':
    main()
