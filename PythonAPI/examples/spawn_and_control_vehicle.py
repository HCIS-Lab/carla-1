import sys
sys.path.append('D:\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.11-py3.7-win-amd64.egg')
sys.path.append('D:\WindowsNoEditor\PythonAPI')
sys.path.append('D:\WindowsNoEditor\PythonAPI\carla')
sys.path.append('D:\WindowsNoEditor\PythonAPI\carla\agents')

import carla
import numpy as np
import time
import csv
import os

client = carla.Client('localhost', 2000)
client.set_timeout(5.0)


'''
Used to draw waypoints

Arguments:
  waypoints: waypoints you want to draw
  road_id: If waypoints are on several roads, you can assign a specific road to draw
  life_time: how long the waypoints shows
'''
def draw_waypoints(waypoints, road_id=None, life_time=50.0): 

  for waypoint in waypoints:

    if(waypoint.road_id == road_id):
      client.get_world().debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                   color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                                   persistent_lines=True)


'''
collect every actors in the map, and output their position and id and time as a csv file

Arguments:
  t: current timestamp
  agent: the object you focus
  filename: the name of csv file
'''
def output_as_csv(t, agent, filename):
  filepath = "D:/WindowsNoEditor/PythonAPI/examples/traj_record/"
  filepath = filepath + filename
  is_exist = os.path.isfile(filepath)
  f = open(filepath, 'a+')
  w = csv.writer(f)

  actors = client.get_world().get_actors()
  town_map = client.get_world().get_map()
  print(town_map.name)
  if not is_exist:
    w.writerow(['TIMESTAMP', 'TRACK_ID', 'OBJECT_TYPE', 'X', 'Y', 'CITY_NAME'])
  
  for actor in actors:
    if actor.type_id[0:7] == 'vehicle' or actor.type_id[0:6] == 'walker':
      x = actor.get_location().x 
      y = actor.get_location().y
      id = actor.id
      if x == agent.get_location().x and y == agent.get_location().y:
        w.writerow([t-10**9, id, 'AGENT', str(x), str(y), town_map.name])
      else:
        w.writerow([t-10**9, id, 'OTHERS', str(x), str(y), town_map.name])



'''
Before running, make sure that every vehicle has been cleaned
'''
def kill_all_vehicle():
  actors = client.get_world().get_actors()
  for actor in actors:
    #print(actor.type_id)
    if actor.type_id[0:7] == 'vehicle':
      actor.destroy()

kill_all_vehicle()

'''
set up the environments
'''
world = client.get_world()
carla_map = world.get_map()

waypoints = client.get_world().get_map().generate_waypoints(distance=1.0)
draw_waypoints(waypoints, road_id=10, life_time=20)
vehicle_blueprint = client.get_world().get_blueprint_library().filter('model3')[0]
filtered_waypoints = []

'''
assign the specific road, set all the waypoints
'''
for waypoint in waypoints:
    if(waypoint.road_id == 10):
      filtered_waypoints.append(waypoint)
spawn_point = filtered_waypoints[-74].transform
spawn_point.location.z += 2
vehicle = client.get_world().spawn_actor(vehicle_blueprint, spawn_point)

from agents.navigation.controller import VehiclePIDController


'''
control the arguments of the car you focus, and draw each waypoint as a green dot on the road
'''
custom_controller = VehiclePIDController(vehicle, args_lateral = {'K_P': 1, 'K_D': 0.0, 'K_I': 0}, args_longitudinal = {'K_P': 1, 'K_D': 0.0, 'K_I': 0.0})
target_waypoint = filtered_waypoints[-2]
client.get_world().debug.draw_string(target_waypoint.transform.location, 'O', draw_shadow=False,
                           color=carla.Color(r=255, g=0, b=0), life_time=20,
                           persistent_lines=True)
waypoint_list = []
for n in range(2, 73, 2):
  waypoint_list.append(filtered_waypoints[n])

ticks_to_track = 20
#for i in range(ticks_to_track):
#	control_signal = custom_controller.run_step(2, target_waypoint)
#	vehicle.apply_control(control_signal)
#print(waypoint_list[-1].transform.location)


'''
Only if the object reach a waypoint, then it can go to next waypoint.
Every 0.1 second, we record all the actors into csv file.
'''
index = 0
time_start = time.time()
while True:    
    #waypoints = carla_map.get_waypoint(vehicle.get_location())
    #waypoint = np.random.choice(waypoints.next(0.3))
    #for target_waypoint in waypoints.next(0.3):
      #client.get_world().debug.draw_string(target_waypoint.transform.location, 'O', draw_shadow=False,
     #                        color=carla.Color(r=0, g=255, b=0), life_time=2,
     #                        persistent_lines=True)
    control_signal = custom_controller.run_step(10, waypoint_list[index])
    vehicle.apply_control(control_signal)

    x_bool = format(vehicle.get_location().x, '.0f') == format(waypoint_list[index].transform.location.x, '.0f')
    y_bool = format(vehicle.get_location().y, '.0f') == format(waypoint_list[index].transform.location.y, '.0f')
    z_bool = format(vehicle.get_location().z, '.0f') == format(waypoint_list[index].transform.location.z, '.0f')

    time_end = time.time()
    if (time_end - time_start) > 0.1:
      #print("time:", time.time(), "x:", vehicle.get_location().x, "y:", vehicle.get_location().y)
      output_as_csv(time.time(), vehicle, '1.csv')
      time_start = time.time()
    
    if x_bool and y_bool and z_bool:
      index += 1
      if index == len(waypoint_list):
        vehicle.destroy()
        break
#while True:
#  if vehicle.get_location() == target_waypoint.transform.location:
#    vehicle.destroy()
#    break

