 ## data_collection.py
 By this script, you can manual control a vehicle and record it by pressing "R". The recorded video contains three positons of RGB camera( left, middle and right) and a LIDAR camera. The angle of the position of the RGB camera can be changed.
 ## manual_control_tt.py
 An optimization version of the original script "manual_control.py". It resolves the lag problem during data recording. 
 ## manual_control_v2.py
 You can print a BEV segmentation image with 500x500 resolutions and save it as a jpg file. We also save all the data after recording with user-defined interval for fear of lagging.
 ## Monitor_control.py
 Recording client's control data for scenario reproduce and automatic data generation. 
 We can control spawn point of agents. Also,we tried to set up a route that all the npc should follow. However, they barely follow the rules.
 ## physics_control.py 
 We changed value in each physical parameters of vehicle to compare the performance of the car we focused.
 ## visualize_multiple_sensors.py
 It is the script that can render multiple sensors in the same pygame window.
 ## reproduce_behavior.py
 Reproduce a scenario with the recorded controll data by assogning scenario name. The file support multple clients' behavior reproducce and random actors generation.
 ## spawn_and_control_vehicle.py
 1. You can use this script to draw waypoints.  
 2. You can make your car follow the user-defined waypoints in sequence.  
 3. It can collect every actors in the map, and output their position and id and time as a csv file. All in all, the output is trajectories of each agents.
 ## data_generator.py
 --random_actors: After specifying this parameter, you can set the center of the scenario, the range of the scenario, and the quantity of vehicles and walkers. 
