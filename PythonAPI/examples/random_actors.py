import sys
sys.path.append('/home/carla/carla/PythonAPI/carla/dist/carla-0.9.11-py3.6-linux-x86_64.egg')
sys.path.append('/home/carla/carla/PythonAPI')
sys.path.append('/home/carla/carla/PythonAPI/carla/')
sys.path.append('/home/carla/carla/PythonAPI/carla/agents')

import carla
import numpy as np
import time
import csv
import os
import logging
import argparse
from numpy import random
from carla import VehicleLightState as vls
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent

client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
def spawn_actor_nearby(center=carla.Location(0, 0, 0), distance=100, vehicle=0, pedestrian=0): 
    # get world and spawn points
    world = client.get_world()
    map = world.get_map()
    spawn_points = map.get_spawn_points()

    """
    topo = map.get_topology()
    topo_dict = {}

    # iterate topology list to create topology dictionary 
    for edge in topo:
        if topo_dict.get(edge[0].id) == None:
            topo_dict[edge[0].id] = []

        topo_dict[edge[0].id].append(edge[1].id)
    """

    #get_spawn_points() get transform
    waypoint_list = []
    for waypoint in spawn_points:
        pt = map.get_waypoint(waypoint.location)
        d = waypoint.location.distance(center)
        closer_node = 0
        next_nodes = pt.next(10)
        #distinguish whether the next waypoint is closer
        num_of_edges = len(next_nodes)
        for node in next_nodes:
            if node.transform.location.distance(center) < d:
                closer_node += 1

        if (waypoint.location.distance(center) < distance and 
            closer_node > num_of_edges//2):
            waypoint_list.append(waypoint)
    
    random.shuffle(waypoint_list)
    print(len(waypoint_list))

    # --------------
    # Spawn vehicles
    # --------------
    num_of_vehicles = 0
    blueprints = world.get_blueprint_library().filter('vehicle.*')
    blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')
    random.seed(int(time.time()))
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor
    traffic_manager = client.get_trafficmanager()
    # keep distance
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)
    synchronous_master = False
    vehicles_list = []
    walkers_list = []
    all_id = []

    batch = []
    for n, transform in enumerate(waypoint_list):
        if n >= vehicle:
            break
        num_of_vehicles += 1
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        # prepare the light state of the cars to spawn
        light_state = vls.NONE
        light_state = vls.Position | vls.LowBeam | vls.LowBeam 

        #client.get_world().spawn_actor(blueprint, transform)
        # spawn the cars and set their autopilot and light state all together
        batch.append(SpawnActor(blueprint, transform)
            .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
            .then(SetVehicleLightState(FutureActor, light_state)))

    for response in client.apply_batch_sync(batch, synchronous_master):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)


    """
    Apply local planner to all vehicle
    Random choose destination and behavior type
    """
    """
    behavior_type = ['aggressive', 'normal', 'cautious']
    actors = client.get_world().get_actors()
    map = client.get_world().get_map()
    vehicles = []
    agents_map = {}

    # set attribute of vehicles
    for actor in actors:
        #print(actor.type_id)
        if actor.type_id[0:7] == 'vehicle':
            randtype = random.choice(behavior_type)
            agent = BehaviorAgent(actor, behavior="aggressive")
            #agent = BasicAgent(actor)

            vehicles.append(actor)
            agents_map[actor] = agent
            
            coord = actor.get_location()
            lane_now = map.get_waypoint(coord).lane_id
            agent.set_destination(coord, coord)

            # sp meams spawn point
            # find destination which in different lane with current location
            random.shuffle(waypoint_list)
            for sp in waypoint_list:
                if map.get_waypoint(sp.location).lane_id != lane_now:
                    agent.set_destination(coord, sp.location)
                    print("Successfully spawn {} actor with initial location {} and destination {}!".format(randtype, coord, sp.location))
                    break
    
    """

    # -------------
    # Spawn Walkers
    # -------------
    # some settings
    percentagePedestriansRunning = 0.5      # how many pedestrians will run
    percentagePedestriansCrossing = 0.5     # how many pedestrians will walk through the road
    # 1. take all the random locations to spawn
    spawn_points = []
    '''
    loc_list = []
    loc_dict = {}
    loc = world.get_random_location_from_navigation()
    count = 1
    
    while loc != None:
        count += 1
        print(loc)
        loc = world.get_random_location_from_navigation()
        if loc_dict.get(loc) == None:
            loc_dict[loc] = True
        else:
            print("overlap")
        loc_list.append(loc)
    print("count:", count) 
    '''
    loc_dict = {}
    for i in range(pedestrian):
        spawn_point = carla.Transform()
        
        flag = False
        
        while True:
        #for j in range(50):
            loc = world.get_random_location_from_navigation()
            temp = carla.Location(int(loc.x), int(loc.y), int(loc.z))
            if (loc.distance(center) < distance) and (loc_dict.get(temp) == None):
                loc_dict[temp] = True
                spawn_point.location = loc
                spawn_points.append(spawn_point)
                flag = True
                break
        #print("Pedestrian#", i, " spawn: ", flag)

        #if (loc != None):
            #spawn_point.location = loc
            #spawn_points.append(spawn_point)
    # 2. we spawn the walker object
    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
    # 4. we put altogether the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)

    # wait for a tick to ensure client receives the last transform of the walkers we have just created
    if not True or not synchronous_master:
        world.wait_for_tick()
    else:
        world.tick()

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point

        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        #all_actors[i].go_to_location(center)

        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

    print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

    # example of how to use parameters
    traffic_manager.global_percentage_speed_difference(30.0)

