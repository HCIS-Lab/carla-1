import json
import os

def write_description(scenario_type, scenario_name ):
    description = scenario_name.split('_')
    topo = description[1].split('-')[0]
    d = dict()

    if 'i' in topo:
        d['topology'] = '4_way_intersection'
    elif 't' in topo:
        if topo[1] == '1':
            d['topology'] = '3_way_intersection_1'
        elif topo[1] == '2':
            d['topology'] = '3_way_intersection_2'
        elif topo[1] == '3':
            d['topology'] = '3_way_intersection_3'
    elif 'r' in topo:
        d['topology'] = 'roundabout'
    elif 's' in topo:
        d['topology'] = 'straight'


    if scenario_type == 'interactive':
        # [topology_id, is_traffic_light, actor_type_action, my_action, violated_rule]
        actor = {'c': 'car', 't': 'truck', 'b': 'bike', 'm': 'motor', 'p': 'pedestrian'}

        action = {'f': 'foward', 'l': 'left_turn', 'r': 'right_turn', 'sl': 'slide_left',
                    'sr': 'slide_right', 'u': 'u-turn', 's': 'stop', 'b': 'backward', 'c': 'crossing',
                    'w': 'walking on sidewalk', 'j': 'jaywalking'}

        violation = {'0': 'None', 'p': 'parking', 'j': 'jay-walker', 'rl': 'running traffic light',
                        's': 'driving on a sidewalk', 'ss': 'stop sign'}

        interaction = {'1': 'True'}
        
        carla_map = {'1':'Town01', '2':'Town02', '3':'Town03', '5':'Town05','6':'Town06', '7':'Town07', '10':'Town10HD' }
        
        
        d['traffic_light'] = 1 if description[2] == '1' else 0
        d['interaction_actor_type'] = actor[description[3]]
        d['interaction_action_type'] = action[description[4]]
        d['my_action'] = action[description[5]]
        d['interaction'] = interaction[description[6]]
        d['violation'] = violation[description[7]]
        d['map'] = carla_map[description[0]]



    elif scenario_type == 'non-interactive':
        # [topology_id, is_traffic_light, actor_type_action, my_action, violated_rule]
        actor = {'0': 'None'}

        action = {'f': 'foward', 'l': 'left_turn', 'r': 'right_turn', 'sl': 'slide_left',
                    'sr': 'slide_right', 'u': 'u-turn', 's': 'stop', 'b': 'backward', 'j': 'jaywalking', '0': 'None'}

        violation = {'0': 'None', 'rl': 'running traffic light',
                        's': 'driving on a sidewalk', 'ss': 'stop sign'}

        interaction = {'0': 'False'}
        carla_map = {'1':'Town01', '2':'Town02', '3':'Town03', '5':'Town05','6':'Town06', '7':'Town07', '10':'Town10HD' }
        
        d['traffic_light'] = 1 if description[2] == '1' else 0
        d['interaction_actor_type'] = actor[description[3]]
        d['interaction_action_type'] = action[description[4]]
        d['my_action'] = action[description[5]]
        d['interaction'] = interaction[description[6]]
        d['violation'] = violation[description[7]]
        d['map'] = carla_map[description[0]]



    elif scenario_type == 'obstacle':
        initial_action = {'f': 'foward', 'l': 'left_turn', 'r': 'right_turn', 'sl': 'slide_left', 'sr': 'slide_right', 
                            'u': 'u-turn'}

        action = {'l': 'left_turn', 'r': 'right_turn', 'sl': 'slide_left', 'sr': 'slide_right', 
                                    'u': 'u-turn'}

        obstacle_type = {'0': 'traffic cone',
                            '1': 'street barrier', '2': 'traffic warning', '3': 'illegal parking'}
        carla_map = {'1':'Town01', '2':'Town02', '3':'Town03', '5':'Town05','6':'Town06', '7':'Town07', '10':'Town10HD' }
        
        d['obstacle type'] = obstacle_type[description[2]]
        d['my_initial_action'] = initial_action[description[3]]
        d['my_action'] = action[description[4]]
        d['map'] = carla_map[description[0]]


    elif scenario_type == 'collision':
        # [topology_id, is_traffic_light, actor_type_action, my_action, violated_rule]

        actor = {'c': 'car', 't': 'truck', 'b': 'bike',
                    'm': 'motor', 'p': 'pedestrian', 's':'static_object'}

        action = {'f': 'foward', 'l': 'left_turn', 'r': 'right_turn', 'sl': 'slide_left',
                    'sr': 'slide_right', 'u': 'u-turn', 's': 'stop', 'b': 'backward', 'c': 'crossing',
                    'w': 'walking on sidewalk', 'j': 'jaywalking', '0': 'None'}

        violation = {'0': 'None', 'p': 'parking', 'j': 'jay-walker', 'rl': 'running traffic light',
                        's': 'driving on a sidewalk', 'ss': 'stop sign'}
        carla_map = {'1':'Town01', '2':'Town02', '3':'Town03', '5':'Town05','6':'Town06', '7':'Town07', '10':'Town10HD' }

        d['traffic_light'] = 1 if description[2] == '1' else 0
        d['interaction_actor_type'] = actor[description[3]]
        d['interaction_action_type'] = action[description[4]]
        d['my_action'] = action[description[5]]
        # d['interaction'] = interaction[description[6]]
        d['violation'] = violation[description[6]]
        d['map'] = carla_map[description[0]]


    with open('./data_collection/'+scenario_type+'/'+scenario_name+'/'+'scenario_description.json', 'w') as f:
        json.dump(d, f)

print("input scenario type\n")
print("1 : interactive\n2 : non-interactive\n3 : obstacle\n4 : collision")
s = int(input())
if (s == 1 ):
    scenario_type = "interactive"
elif(s == 2 ):
    scenario_type = "non-interactive"
elif(s == 3 ):
    scenario_type = "obstacle"
elif(s == 4 ):
    scenario_type = "collision"
else:
    scenario_type = "interactive"

list = os.listdir('./data_collection/'+scenario_type)

for scenario_name in list:
    write_description(scenario_type, scenario_name )