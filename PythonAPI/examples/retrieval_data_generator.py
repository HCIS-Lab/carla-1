#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.


from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import random
import csv
import json
from turtle import back

try:
    #
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    #
    # sys.path.append('../carla/agents/navigation')
    sys.path.append('../carla/agents')
    sys.path.append('../carla/')
    sys.path.append('../../HDMaps')
    sys.path.append('rss/') # rss

except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla
from carla import VehicleLightState as vls

from carla import ColorConverter as cc
from carla import Transform, Location, Rotation
from controller import VehiclePIDController
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import time
import threading
from multiprocessing import Process

import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from random_actors import spawn_actor_nearby
from get_and_control_trafficlight import *
from read_input import *
# rss
# from rss_sensor_benchmark import RssSensor # pylint: disable=relative-import
# from rss_visualization import RssUnstructuredSceneVisualizer, RssBoundingBoxVisualizer, RssStateVisualizer # pylint: disable=relative-import
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_e
    from pygame.locals import K_o
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters)
               if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def write_json(filename, index, seed ):
    with open(filename,'r+') as file:
          # First we load existing data into a dict.
        file_data = json.load(file)
        y = {str(index):seed}
        file_data.update(y)
        file.seek(0)
        json.dump(file_data, file, indent = 4)

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, client_bp, hud, args, store_path):
        self.world = carla_world
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.abandon_scenario = False
        self.finish = False
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True  # Enables synchronous mode
        self.world.apply_settings(settings)
        self.actor_role_name = args.rolename
        self.store_path = store_path
        self.args = args
        
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print(
                '  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = client_bp
        self._gamma = args.gamma
        self.ego_data = {}
        self.save_mode = not args.no_save
        self.restart(self.args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]
        
    def restart(self, args):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        
        seed_1 = int(time.time())

        d = {"1": seed_1}
        # if args.replay:
        #     #print(self.store_path)
        #     P = self.store_path.split("/")
        #     #print(P)
        #     with open(os.path.join('data_collection', args.scenario_type,  args.scenario_id, 'variant_scenario', P[3] )+"/random_seeds.json", "r") as outfile:
        #         data = json.load(outfile)
        #         seed_1 = int(data["1"])
        # else:
        #     with open(self.store_path + "/random_seeds.json", "w+") as outfile:
        #         json.dump(d, outfile)
        # print("seed_1: ", seed_1)
        random.seed(seed_1)


        blueprint = random.choice(
            self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            seed_2 = int(time.time()) + 20

            # if args.replay:
            #     P = self.store_path.split("/")
            #     #print(P)
            #     with open(os.path.join('data_collection', args.scenario_type,  args.scenario_id, 'variant_scenario', P[3] )+"/random_seeds.json", "r") as outfile:
            
            #         data = json.load(outfile)
            #         seed_2 = int(data["2"])
            # else:

            #     write_json(self.store_path + "/random_seeds.json", 2, seed_2 )
            # print("seed_2: ", seed_2)
            random.seed(seed_2)
                    
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        if blueprint.has_attribute('driver_id'):
            # if args.replay:
            #     P = self.store_path.split("/")
            #     #print(P)
            #     with open(os.path.join('data_collection', args.scenario_type,  args.scenario_id, 'variant_scenario', P[3] )+"/random_seeds.json", "r") as outfile:
            #         data = json.load(outfile)
            #         seed_3 = int(data["3"])
            # else:
            #     seed_3 = int(time.time()) + int( random.random())
            #     write_json(self.store_path + "/random_seeds.json", 3, seed_3 )
            # print("seed_3: ", seed_3)
            random.seed(seed_3)
            
            driver_id = random.choice(
                blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(
                blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(
                blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(
                spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            # coor=carla.Location(location[0],location[1],location[2]+2.0)
            # self.player.set_location(coor)
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player, self.ego_data)
        self.imu_sensor = IMUSensor(self.player, self.ego_data)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma, self.save_mode)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.camera_manager.background = True
        self.camera_manager.save_mode = self.save_mode

        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        # rss
        if self.args.save_rss:
            self.rss_unstructured_scene_visualizer = RssUnstructuredSceneVisualizer(self.player, self.world, self.hud.dim)
            self.rss_bounding_box_visualizer = RssBoundingBoxVisualizer(self.hud.dim, self.world, self.camera_manager.sensor_top)
            self.rss_sensor = RssSensor(self.player, self.world,
                                        self.rss_unstructured_scene_visualizer, self.rss_bounding_box_visualizer, self.hud.rss_state_visualizer)
        # rss end

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        # If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        self.hud.tick(self, clock)

    def record_speed_control_transform(self, frame):
        v = self.player.get_velocity()
        c = self.player.get_control()
        t = self.player.get_transform()
        if frame not in self.ego_data:
            self.ego_data[frame] = {}
        self.ego_data[frame]['speed'] = {'constant': math.sqrt(v.x**2 + v.y**2 + v.z**2),
                                         'x': v.x, 'y': v.y, 'z': v.z}
        self.ego_data[frame]['control'] = {'throttle': c.throttle, 'steer': c.steer,
                                           'brake': c.brake, 'hand_brake': c.hand_brake,
                                           'manual_gear_shift': c.manual_gear_shift,
                                           'gear': c.gear}
        self.ego_data[frame]['transform'] = {'x': t.location.x, 'y': t.location.y, 'z': t.location.z,
                                             'pitch': t.rotation.pitch, 'yaw': t.rotation.yaw, 'roll': t.rotation.roll}
    def save_ego_data(self, path):
        self.imu_sensor.toggle_recording_IMU()
        self.gnss_sensor.toggle_recording_Gnss()
        with open(os.path.join(path, 'ego_data.json'), 'w') as f:
            json.dump(self.ego_data, f, indent=4)
        self.ego_data = {}

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        if self.save_mode:
            sensors = [
                # self.camera_manager.sensor_lbc_img,
                self.camera_manager.sensor_front,
                self.camera_manager.sensor_lbc_ins,
                self.camera_manager.ins_front,
                self.camera_manager.sensor_flow,
                self.camera_manager.depth_front,
                self.camera_manager.sensor_lidar,
                self.collision_sensor.sensor,
                self.lane_invasion_sensor.sensor,
                self.gnss_sensor.sensor,
                self.imu_sensor.sensor]

            self.camera_manager.sensor_front = None


        else:
            sensors = [
            # self.camera_manager.lbc_img,
            self.camera_manager.lbc_ins,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]

        if self.args.save_rss and self.save_mode:
            # rss
            if self.rss_sensor:
                self.rss_sensor.destroy()
            if self.rss_unstructured_scene_visualizer:
                self.rss_unstructured_scene_visualizer.destroy()


        for i, sensor in enumerate(sensors):
            if sensor is not None:
                sensor.stop()
                sensor.destroy()


        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = None
            self.control_list = None
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        r = 2
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 1
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return 1
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_e:
                    xx = int(input("x: "))
                    yy = int(input("y: "))
                    zz = int(input("z: "))
                    new_location = carla.Location(xx, yy, zz)
                    world.player.set_location(new_location)
                elif event.key == K_o:
                    xyz = [float(s) for s in input(
                        'Enter coordinate: x , y , z  : ').split()]
                    new_location = carla.Location(xyz[0], xyz[1], xyz[2])
                    world.player.set_location(new_location)
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification(
                            "Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(
                            carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification(
                            "Enabled Constant Velocity Mode at 60 km/h")
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    scenario_name = None
                    world.camera_manager.recording = not world.camera_manager.recording
                    # world.lidar_sensor.recording= not  world.lidar_sensor.recording
                    # if not  world.lidar_sensor.recording:
                    if not world.camera_manager.recording:
                        scenario_name = input("scenario id: ")
                    world.camera_manager.toggle_recording(scenario_name)

                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")

                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification(
                        "Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec",
                                       world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification(
                        "Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification(
                        "Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker
                if event.key == K_r:
                    r = 3

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(
                    pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else:  # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else:  # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights:  # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(
                        carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(
                    pygame.key.get_pressed(), clock.get_time(), world)
            # world.player.apply_control(self._control)
            return 0

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods(
            ) & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height, world, args):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        # self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self._world = world
        self.args = args
        self.frame = 0
        if self.args.save_rss: # rss
            self._world = world 
            self.rss_state_visualizer = RssStateVisualizer(self.dim, self._font_mono, self._world)

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(
                world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(
                seconds=int(self.simulation_time)),
            'Frame:   %s' % self.frame,
            '',
            'Speed:   % 15.0f km/h' % (3.6 *
                                       math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            #'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            #'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            #'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' %
                                (t.location.x, t.location.y)),
            #'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        # self._info_text += [
        #     '',
        #     'Collision:',
        #     collision,
        #     '',
        #     'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            def distance(l): return math.sqrt((l.x - t.location.x) **
                                              2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x)
                        for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(
                            display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255),
                                         rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(
                        item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
            if self.args.save_rss:
                self.rss_state_visualizer.render(display, v_offset) # rss
        self._notifications.render(display)
        # self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""

    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 *
                    self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        self.other_actor_id = 0 # init as 0 for static object
        self.wrong_collision = False
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))
        self.collision = False

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        self.history.append({'frame': event.frame, 'actor_id': event.other_actor.id})
        # if len(self.history) > 4000:
        #     self.history.pop(0)
        self.collision = True
        if event.other_actor.id != self.other_actor_id:
            self.wrong_collision = True
        

    def save_history(self, path):
        if self.collision:
            # for i, collision in enumerate(self.history):
            #     self.history[i] = list(self.history[i])
            # history = np.asarray(self.history)
            # if len(history) != 0:
            #     np.save('%s/collision_history' % (path), history)
            with open(os.path.join(path, 'collision_history.json'), 'w') as f:
                json.dump(self.history, f, indent=4)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor, ego_data):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(
            carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        self.recording = False
        self.ego_dict = ego_data
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude
        if self.recording:
            gnss = {'lat': event.latitude, 'lon': event.longitude}
            gnss_transform = {'x': event.transform.location.x, 'y': event.transform.location.y, 'z': event.transform.location.z,
                              'pitch': event.transform.rotation.pitch, 'yaw': event.transform.rotation.yaw, 'roll': event.transform.rotation.roll}

            if not event.frame in self.ego_dict:
                self.ego_dict[event.frame] = {}
            self.ego_dict[event.frame]['gnss'] = gnss
            self.ego_dict[event.frame]['gnss_transform'] = gnss_transform

    def toggle_recording_Gnss(self):
        self.recording = not self.recording

# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor, ego_data):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.recording = False
        # self.imu_save = []
        self.ego_dict = ego_data
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(
                sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(
                sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)
        if self.recording:
            imu = {'accelerometer_x': self.accelerometer[0], 'accelerometer_y': self.accelerometer[1],
                   'accelerometer_z': self.accelerometer[2], 'gyroscope_x': self.gyroscope[0],
                   'gyroscope_y': self.gyroscope[1], 'gyroscope_z': self.gyroscope[2],
                   'compass': self.compass}

            if not sensor_data.frame in self.ego_dict:
                self.ego_dict[sensor_data.frame] = {}
            self.ego_dict[sensor_data.frame]['imu'] = imu
            self.ego_dict[sensor_data.frame]['timestamp'] = sensor_data.timestamp
    def toggle_recording_IMU(self):
        self.recording = not self.recording

# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5  # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=2.8, z=1.0),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / \
                self.velocity_range  # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction, save_mode):
        self.sensor_front = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.save_mode = save_mode

        # self.lbc_img = []
        self.lbc_ins = []
        self.front_img = []
        self.front_ins = []
        self.flow = []
        self.lidar = []
        self.front_depth = []

        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                # front view
                (carla.Transform(carla.Location(x=+0.8*bound_x,
                 y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                # front-left view
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=-55)), Attachment.Rigid),
                # front-right view
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=55)), Attachment.Rigid),
                # back view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=180)), Attachment.Rigid),
                # back-left view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=235)), Attachment.Rigid),
                # back-right view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=-235)), Attachment.Rigid),
                # top view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                 z=23*bound_z), carla.Rotation(pitch=18.0)), Attachment.SpringArm),
                # LBC top view
                (carla.Transform(carla.Location(x=0, y=0,
                 z=100.0), carla.Rotation(pitch=-90.0)), Attachment.SpringArm)
            ]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-5.5, z=2.5),
                 carla.Rotation(pitch=8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)),
                 Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-8.0, z=6.0),
                 carla.Rotation(pitch=6.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth,
                'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw,
                'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None,
                'Lidar (Ray-Cast)', {'range': '85', 'rotation_frequency': '25'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.optical_flow', None, 'Optical Flow', {}],
            ['sensor.other.lane_invasion', None, 'Lane lane_invasion', {}],
            ['sensor.camera.instance_segmentation', cc.CityScapesPalette,
                'Camera Instance Segmentation (CityScapes Palette)', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()

        # self.bev_bp = bp_library.find('sensor.camera.rgb')
        # self.bev_bp.set_attribute('image_size_x', str(300))
        # self.bev_bp.set_attribute('image_size_y', str(300))
        # self.bev_bp.set_attribute('fov', str(50.0))
        # if self.bev_bp.has_attribute('gamma'):
        #     self.bev_bp.set_attribute('gamma', str(gamma_correction))


        self.bev_seg_bp = bp_library.find('sensor.camera.instance_segmentation')
        self.bev_seg_bp.set_attribute('image_size_x', str(400))
        self.bev_seg_bp.set_attribute('image_size_y', str(400))
        self.bev_seg_bp.set_attribute('fov', str(50.0))

        self.front_cam_bp = bp_library.find('sensor.camera.rgb')
        self.front_cam_bp.set_attribute('image_size_x', str(1536))
        self.front_cam_bp.set_attribute('image_size_y', str(512))
        self.front_cam_bp.set_attribute('fov', str(120.0))
        self.front_cam_bp.set_attribute('lens_circle_multiplier', '0.0')
        self.front_cam_bp.set_attribute('lens_circle_falloff', '0.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_intensity', '3.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_offset', '500')
        # self.front_cam_bp.set_attribute('focal_distance', str(500))
        if self.front_cam_bp.has_attribute('gamma'):
            self.front_cam_bp.set_attribute('gamma', str(gamma_correction))

        self.front_seg_bp = bp_library.find('sensor.camera.instance_segmentation')
        self.front_seg_bp.set_attribute('image_size_x', str(1536))
        self.front_seg_bp.set_attribute('image_size_y', str(512))
        self.front_seg_bp.set_attribute('fov', str(120.0))
        self.front_cam_bp.set_attribute('lens_circle_multiplier', '0.0')
        self.front_cam_bp.set_attribute('lens_circle_falloff', '0.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_intensity', '3.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_offset', '500')
        # self.front_seg_bp.set_attribute('focal_distance', str(500))
        if self.front_seg_bp.has_attribute('gamma'):
            self.front_seg_bp.set_attribute('gamma', str(gamma_correction))

        self.depth_bp = bp_library.find('sensor.camera.depth')
        self.depth_bp.set_attribute('image_size_x', str(1536))
        self.depth_bp.set_attribute('image_size_y', str(512))
        self.depth_bp.set_attribute('fov', str(120.0))
        self.front_cam_bp.set_attribute('lens_circle_multiplier', '0.0')
        self.front_cam_bp.set_attribute('lens_circle_falloff', '0.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_intensity', '3.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_offset', '500')
        # self.depth_bp.set_attribute('focal_distance', str(500))
        if self.depth_bp.has_attribute('gamma'):
            self.depth_bp.set_attribute('gamma', str(gamma_correction))

        self.flow_bp = bp_library.find('sensor.camera.optical_flow')
        self.flow_bp.set_attribute('image_size_x', str(1236))
        self.flow_bp.set_attribute('image_size_y', str(512))
        self.flow_bp.set_attribute('fov', str(120.0))
        self.front_cam_bp.set_attribute('lens_circle_multiplier', '0.0')
        self.front_cam_bp.set_attribute('lens_circle_falloff', '0.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_intensity', '3.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_offset', '500')
        # self.flow_bp.set_attribute('focal_distance', str(500))
        if self.flow_bp.has_attribute('gamma'):
            self.flow_bp.set_attribute('gamma', str(gamma_correction))
        for item in self.sensors:
            
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index +
                                1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index]
             [2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor_front is not None:
                self.sensor_front.destroy()
                self.surface = None

            # rgb sensor
            if self.save_mode:
                self.sensor_front = self._parent.get_world().spawn_actor(
                    self.front_cam_bp,
                    self._camera_transforms[0][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])
                self.sensor_lbc_ins = self._parent.get_world().spawn_actor(
                    self.bev_seg_bp,
                    self._camera_transforms[7][0],
                    attach_to=self._parent)
                # self.sensor_lbc_img = self._parent.get_world().spawn_actor(
                #     self.bev_bp,
                #     self._camera_transforms[7][0],
                #     attach_to=self._parent)
                self.ins_front = self._parent.get_world().spawn_actor(
                    self.front_seg_bp,
                    self._camera_transforms[0][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])
                # depth estimation sensor
                self.depth_front = self._parent.get_world().spawn_actor(
                    self.depth_bp,
                    self._camera_transforms[0][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])
                self.sensor_flow = self._parent.get_world().spawn_actor(
                    self.flow_bp,
                    self._camera_transforms[0][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])
                self.sensor_lidar = self._parent.get_world().spawn_actor(
                    self.sensors[6][-1],
                    self._camera_transforms[0][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            # self.sensor_lbc_img.listen(
            #     lambda image: CameraManager._parse_image(weak_self, image, 'lbc_img'))

            if self.save_mode:
                self.sensor_front.listen(lambda image: CameraManager._parse_image(weak_self, image, 'front'))
                self.sensor_lbc_ins.listen(lambda image: CameraManager._parse_image(weak_self, image, 'lbc_ins'))
                self.ins_front.listen(lambda image: CameraManager._parse_image(weak_self, image, 'ins_front'))
                self.depth_front.listen(lambda image: CameraManager._parse_image(weak_self, image, 'depth_front'))
                self.sensor_flow.listen(lambda image: CameraManager._parse_image(weak_self, image, 'flow'))
                self.sensor_lidar.listen(lambda image: CameraManager._parse_image(weak_self, image, 'lidar'))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self, path):
        self.recording = not self.recording
        if not self.recording:
            # t_lbc_img = Process(target=self.save_img, args=(self.lbc_img, 0, path, 'lbc_img'))
            t_front = Process(target=self.save_img, args=(self.front_img, 0, path, 'front'))
            t_lbc_ins = Process(target=self.save_img, args=(self.lbc_ins, 10, path, 'lbc_ins'))
            t_ins_front = Process(target=self.save_img, args=(self.front_ins, 10, path, 'ins_front'))
            t_depth_front = Process(target=self.save_img, args=(self.front_depth, 1, path, 'depth_front'))
            t_flow = Process(target=self.save_img, args=(self.flow, 8, path, 'flow'))
            t_lidar = Process(target=self.save_img, args=(self.lidar, 6, path, 'lidar'))
            start_time = time.time()

            # t_lbc_img.start()
            t_front.start()
            t_lbc_ins.start()
            t_ins_front.start()
            t_depth_front.start()
            t_flow.start()
            t_lidar.start()
            
            # t_lbc_img.join()
            t_front.join()
            t_lbc_ins.join()
            t_ins_front.join()
            t_depth_front.join()
            t_flow.join()
            t_lidar.join()
            
            # self.lbc_img = []
            self.lbc_ins = []
            self.front_img = []
            self.front_ins = []
            self.front_depth = []
            self.flow = []
            self.lidar = []

            end_time = time.time()
            print('sensor data save done in %s' % (end_time-start_time))
        self.hud.notification('Recording %s' %
                              ('On' if self.recording else 'Off'))

    def save_img(self, img_list, sensor, path, view='top'):
        modality = self.sensors[sensor][0].split('.')[-1]
        for img in img_list:
            if img.frame % 1 == 0:
                if 'seg' in view:
                    img.save_to_disk(
                        '%s/%s/%s/%08d' % (path, modality, view, img.frame), cc.CityScapesPalette)
                elif 'depth' in view:
                    img.save_to_disk(
                        '%s/%s/%s/%08d' % (path, modality, view, img.frame), cc.LogarithmicDepth)
                elif 'flow' in view:
                    frame = img.frame
                    img = img.get_color_coded_flow()
                    array = np.frombuffer(
                        img.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (img.height, img.width, 4))
                    array = array[:, :, :3]
                    array = array[:, :, ::-1]
                    stored_path = os.path.join(path, modality, view)
                    if not os.path.exists(stored_path):
                        os.makedirs(stored_path)
                    np.save('%s/%08d' % (stored_path, frame), array)
                else:
                    img.save_to_disk('%s/%s/%s/%08d' %
                                     (path, modality, view, img.frame))
        print("%s %s save finished." % (self.sensors[sensor][2], view))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image, view='top'):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)

        elif view == 'front':
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            # render the view shown in monitor
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        if self.recording and image.frame % 1 == 0:
            # print(view,image.frame)
            if view == 'top':
                self.top_img.append(image)
            # elif view == 'lbc_img':
            #     self.lbc_img.append(image)
            elif view == 'front':
                self.front_img.append(image)
            elif view == 'lbc_ins':
                self.lbc_ins.append(image)
            elif view == 'lidar':
                self.lidar.append(image)
            elif view == 'flow':
                self.flow.append(image)
            elif view == 'seg_front':
                self.front_seg.append(image)
            elif view == 'ins_front':
                self.front_ins.append(image)

            elif view == 'depth_front':
                self.front_depth.append(image)


def record_control(control, control_list):
    np_control = np.zeros(7)
    np_control[0] = control.throttle
    np_control[1] = control.steer
    np_control[2] = control.brake
    np_control[3] = control.hand_brake
    np_control[4] = control.reverse
    np_control[5] = control.manual_gear_shift
    np_control[6] = control.gear

    control_list.append(np_control)


def control_with_trasform_controller(controller, transform):
    control_signal = controller.run_step(10, transform)
    return control_signal



def set_bp(blueprint, actor_id):
    blueprint = random.choice(blueprint)
    blueprint.set_attribute('role_name', 'tp')
    if blueprint.has_attribute('color'):
        color = random.choice(
            blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)
    if blueprint.has_attribute('driver_id'):
        driver_id = random.choice(
            blueprint.get_attribute('driver_id').recommended_values)
        blueprint.set_attribute('driver_id', driver_id)
    if blueprint.has_attribute('is_invincible'):
        blueprint.set_attribute('is_invincible', 'true')


    # set the max speed
    # if blueprint.has_attribute('speed'):
    #     self.player_max_speed = float(
    #         blueprint.get_attribute('speed').recommended_values[1])
    #     self.player_max_speed_fast = float(
    #         blueprint.get_attribute('speed').recommended_values[2])
    # else:
    #     print("No recommended values for 'speed' attribute")
    return blueprint


def save_description(world, args, stored_path, weather,agents_dict, nearest_obstacle):
    vehicles = world.world.get_actors().filter('vehicle.*')
    peds = world.world.get_actors().filter('walker.*')
    d = dict()
    d['num_actor'] = len(vehicles) + len(peds)
    d['num_vehicle'] = len(vehicles)
    d['weather'] = str(weather)
    # d['random_objects'] = args.random_objects
    d['random_actors'] = args.random_actors
    d['simulation_time'] = int(world.hud.simulation_time)
    d['nearest_obstacle'] = nearest_obstacle
    
    for key in agents_dict:
        d[key] = agents_dict[key].id

    with open('%s/dynamic_description.json' % (stored_path), 'w') as f:
        json.dump(d, f)


def write_actor_list(world,stored_path):

    def write_row(writer,actors,filter_str,class_id,min_id,max_id):
        filter_actors = actors.filter(filter_str)
        for actor in filter_actors:
            if actor.id < min_id:
                min_id = actor.id
            if actor.id > max_id:
                max_id = actor.id
            writer.writerow([actor.id,class_id,actor.type_id])
        return min_id,max_id
    
    filter_ = ['walker.*','vehicle.*','static.prop.streetbarrier*',
            'static.prop.trafficcone*','static.prop.trafficwarning*']
    id_ = [4,10,20,20,20]
    actors = world.world.get_actors()
    min_id = int(1e7)
    max_id = int(0)
    with open(stored_path+'/actor_list.csv', 'w') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(['Actor_ID','Class','Blueprint'])
        for filter_str,class_id in zip(filter_,id_):
            min_id, max_id = write_row(writer,actors,filter_str,class_id,min_id,max_id)
        print('min id: {}, max id: {}'.format(min_id,max_id))
    return min_id,max_id

def generate_obstacle(world, bp, path, ego_transform):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    min_dis = float('Inf')
    nearest_obstacle = -1
    
    if lines[0][:6] == 'static':
        for line in lines:
            obstacle_name = line.split('\t')[0]
            transform = line.split('\t')[1]
            # print(obstacle_name, " ", transform)
            # exec("obstacle_actor = world.spawn_actor(bp.filter(obstacle_name)[0], %s)" % transform)

            x = float(transform.split('x=')[1].split(',')[0])
            y = float(transform.split('y=')[1].split(',')[0])
            z = float(transform.split('z=')[1].split(')')[0])
            pitch = float(transform.split('pitch=')[1].split(',')[0])
            yaw = float(transform.split('yaw=')[1].split(',')[0])
            roll = float(transform.split('roll=')[1].split(')')[0])

            obstacle_loc = carla.Location(x, y, z)
            obstacle_rot = carla.Rotation(pitch, yaw, roll)
            obstacle_trans = carla.Transform(obstacle_loc, obstacle_rot)

            obstacle_actor = world.spawn_actor(bp.filter(obstacle_name)[0], obstacle_trans)

            dis = ego_transform.location.distance(obstacle_loc)
            if dis < min_dis:
                nearest_obstacle = obstacle_actor.id
                min_dis = dis

    return nearest_obstacle

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================
import cv2

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    if args.map == 'Town07':
        return

    if args.map == 'Town03' or args.map == 'Town05':
        max_folder = 25
    else:
        max_folder = 15

    stored_path = os.path.join('data_collection', args.scenario_type, args.scenario_id, 'variant_scenario')
    if not os.path.exists(stored_path) :
        os.makedirs(stored_path)
    folders = [f for f in os.listdir(stored_path) if os.path.isdir(os.path.join(stored_path, f))]
    if len(folders) == 0:
        v_id = str(1)
    elif len(folders) > max_folder:
        return
    else:
        for i in range(len(folders)):
            folders[i] = int(folders[i])
        v_id = str(int(max(folders))+1)
    stored_path = os.path.join(stored_path, v_id)
    if not os.path.exists(stored_path) :
        os.makedirs(stored_path)
    print(stored_path)

    path = os.path.join('data_collection', args.scenario_type, args.scenario_id)

    out = cv2.VideoWriter(stored_path+"/"+v_id+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20,  (1536, 512) )

    filter_dict = {}
    try:
        for root, _, files in os.walk(path + '/filter/'):
            for name in files:
                f = open(path + '/filter/' + name, 'r')
                bp = f.readlines()[0]
                bp = bp.strip('\n')
                name = name.strip('.txt')
                f.close()
                filter_dict[name] = bp
        print(filter_dict)
    except:
        print("檔案夾不存在。")

    # load files for scenario reproducing
    transform_dict = {}
    velocity_dict = {}
    ped_control_dict = {}
    for actor_id, filter in filter_dict.items():
        transform_dict[actor_id] = read_transform(
            os.path.join(path, 'transform', actor_id + '.npy'))
        velocity_dict[actor_id] = read_velocity(
            os.path.join(path, 'velocity', actor_id + '.npy'))
        if 'pedestrian' in filter:
            ped_control_dict[actor_id] = read_ped_control(
                os.path.join(path, 'ped_control', actor_id + '.npy'))
    # num_files = len(filter_dict)
    abandon_scenario = False
    # stored_path = None
    scenario_name = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0, 0, 0))
        pygame.display.flip()

        hud = HUD(args.width, args.height, client.get_world(), args)

        weather = args.weather
        exec("args.weather = carla.WeatherParameters.%s" % args.weather)
        
        world = World(client.load_world(args.map),
                      filter_dict['player'], hud, args, stored_path)
        client.get_world().set_weather(args.weather)
        # sync mode
        settings = world.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True  # Enables synchronous mode
        world.world.apply_settings(settings)
        # other setting
        controller = KeyboardControl(world, args.autopilot)
        blueprint_library = client.get_world().get_blueprint_library()


        lights = []
        actors = world.world.get_actors().filter('traffic.traffic_light*')
        for l in actors:
            lights.append(l)
        light_dict, light_transform_dict = read_traffic_lights(path, lights)
        
        clock = pygame.time.Clock()

        agents_dict = {}
        controller_dict = {}
        actor_transform_index = {}
        finish = {}

        # init position for player
        ego_transform = transform_dict['player'][0]
        ego_transform.location.z += 3
        world.player.set_transform(ego_transform)
        agents_dict['player'] = world.player
        

        # generate obstacles and calculate the distance between ego-car and nearest obstacle
        min_dis = float('Inf')
        nearest_obstacle = -1
        if args.scenario_type == 'obstacle':
            nearest_obstacle = generate_obstacle(client.get_world(), blueprint_library,
                              path+"/obstacle/obstacle_list.txt", ego_transform)


        # set controller
        for actor_id, bp in filter_dict.items():
            if actor_id != 'player':
                transform_spawn = transform_dict[actor_id][0]
                
                while True:
                    try:
                        agents_dict[actor_id] = client.get_world().spawn_actor(
                            set_bp(blueprint_library.filter(
                                filter_dict[actor_id]), actor_id),
                            transform_spawn)
                        print(filter_dict[actor_id])

                        if args.scenario_type == 'obstacle':
                            dis = ego_transform.location.distance(transform_spawn.location)
                            if dis < min_dis:
                                nearest_obstacle = actor_id
                                min_dis = dis

                        break
                    except Exception:
                        transform_spawn.location.z += 1.5

                # set other actor id for checking collision object's identity
                world.collision_sensor.other_actor_id = agents_dict[actor_id].id

            if 'vehicle' in bp:
                controller_dict[actor_id] = VehiclePIDController(agents_dict[actor_id], args_lateral={'K_P': 1, 'K_D': 0.0, 'K_I': 0}, args_longitudinal={'K_P': 1, 'K_D': 0.0, 'K_I': 0.0},
                                                                    max_throttle=1.0, max_brake=1.0, max_steering=1.0)
                try:
                    agents_dict[actor_id].set_light_state(carla.VehicleLightState.LowBeam)
                except:
                    print('vehicle has no low beam light')
            actor_transform_index[actor_id] = 1
            finish[actor_id] = False

        root = os.path.join('data_collection', args.scenario_type, args.scenario_id)
        scenario_name = str(weather) + '_'

        vehicles_list = []
        all_id = None 
        if args.random_actors != 'none':
            if args.random_actors == 'pedestrian':  #only pedestrian
                vehicles_list, all_actors,all_id = spawn_actor_nearby(args, stored_path, distance=100, v_ratio=0.0,
                                   pedestrian=40 , transform_dict=transform_dict)
            elif args.random_actors == 'low':
                vehicles_list, all_actors,all_id = spawn_actor_nearby(args, stored_path, distance=100, v_ratio=0.3,
                                   pedestrian=20 , transform_dict=transform_dict)
            elif args.random_actors == 'mid':
                vehicles_list, all_actors,all_id = spawn_actor_nearby(args, stored_path, distance=100, v_ratio=0.6,
                                   pedestrian=40, transform_dict=transform_dict)
            elif args.random_actors == 'high':
                vehicles_list, all_actors,all_id = spawn_actor_nearby(args, stored_path, distance=100, v_ratio=0.8,
                                   pedestrian=80, transform_dict=transform_dict)
        scenario_name = scenario_name + args.random_actors + '_'

        if not args.no_save:
            # recording traj
            id = []
            moment = []
            print(root)
            with open(os.path.join(root, 'timestamp.txt')) as f:
                for line in f.readlines():
                    s = line.split(',')
                    id.append(int(s[0]))
                    moment.append(s[1])
            period = float(moment[-1]) - float(moment[0])
            half_period = period / 2

        if args.save_rss:
            print(world.rss_sensor)
            world.rss_sensor.stored_path = stored_path
        # write actor list

        min_id, max_id = write_actor_list(world,stored_path)
        if max_id-min_id>=65535:
            print('Actor id error. Abandom.')
            abandon_scenario = True
            raise 

        iter_tick = 0
        iter_start = 25
        iter_toggle = 50

        while (1):
            clock.tick_busy_loop(40)
            frame = world.world.tick()
            hud.frame = frame
            iter_tick += 1
            if iter_tick == iter_start + 1:
                ref_light = get_next_traffic_light(
                    world.player, world.world, light_transform_dict)
                annotate = annotate_trafficlight_in_group(
                    ref_light, lights, world.world)
            
            elif iter_tick > iter_start:
                                
                # iterate actors
                for actor_id, _ in filter_dict.items():
                    # apply recorded location and velocity on the controller
                    actors = world.world.get_actors()
                    # reproduce traffic light state
                    if actor_id == 'player' and ref_light:
                        set_light_state(
                            lights, light_dict, actor_transform_index[actor_id], annotate)

                    if actor_transform_index[actor_id] < len(transform_dict[actor_id]):
                        x = transform_dict[actor_id][actor_transform_index[actor_id]].location.x
                        y = transform_dict[actor_id][actor_transform_index[actor_id]].location.y
                        if 'vehicle' in filter_dict[actor_id]:

                            target_speed = (velocity_dict[actor_id][actor_transform_index[actor_id]])*3.6
                            waypoint = transform_dict[actor_id][actor_transform_index[actor_id]]

                            agents_dict[actor_id].apply_control(controller_dict[actor_id].run_step(target_speed, waypoint))                            

                            v = agents_dict[actor_id].get_velocity()
                            v = ((v.x)**2 + (v.y)**2+(v.z)**2)**(0.5)

                            # to avoid the actor slowing down for the dense location around
                            # if agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) < 2 + v/20.0:
                            if agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) < 2.0:
                                actor_transform_index[actor_id] += 2
                            elif agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) > 6.0:
                                actor_transform_index[actor_id] += 6
                            else:
                                actor_transform_index[actor_id] += 1

                            if actor_id == 'player' and not args.no_save:
                                world.record_speed_control_transform(frame)
                                

                        elif 'pedestrian' in filter_dict[actor_id]:
                            agents_dict[actor_id].apply_control(
                                ped_control_dict[actor_id][actor_transform_index[actor_id]])
                            actor_transform_index[actor_id] += 1

                            # w.writerow(
                            #         [frame, actor_id, 'actor.pedestrian', str(x), str(y), args.map])
                    else:
                        finish[actor_id] = True

                        # elif actor_id == 'player':
                        #     scenario_finished = True
                        #     break
                # for actor in actors:
                #     if actor_transform_index['player'] < len(transform_dict[actor_id]):
                #         if actor.type_id[0:7] == 'vehicle' or actor.type_id[0:6] == 'walker':
                #             x = actor.get_location().x
                #             y = actor.get_location().y
                #             id = actor.id
                            # if actor.type_id[0:7] == 'vehicle':
                            #     w.writerow(
                            #             [frame, id, 'vehicle', str(x), str(y), args.map])
                            # elif actor.type_id[0:6] == 'walker':
                            #     w.writerow(
                            #             [frame, id, 'pedestrian', str(x), str(y), args.map])
                if not False in finish.values():
                    break

                if controller.parse_events(client, world, clock) == 1:
                    return

                if world.collision_sensor.collision and args.scenario_type != 'collision':
                    print('unintentional collision, abandon scenario')
                    abandon_scenario = True
                elif world.collision_sensor.wrong_collision:
                    print('collided with wrong object, abandon scenario')
                    abandon_scenario = True

                if abandon_scenario:
                    world.abandon_scenario = True
                    break
            if iter_tick == iter_toggle:
                if not args.no_save:
                    time.sleep(3)
                    world.camera_manager.toggle_recording(stored_path)
                    world.imu_sensor.toggle_recording_IMU()
                    world.gnss_sensor.toggle_recording_Gnss()

                    # start recording .log file
                    # print("Recording on file: %s" % client.start_recorder(os.path.join(os.path.abspath(os.getcwd()), stored_path, 'recording.log'),True))
            elif iter_tick > iter_toggle:
                pygame.image.save(display, os.path.join(stored_path, "screenshot.jpeg"))
                image = cv2.imread(os.path.join(stored_path, "screenshot.jpeg"))
                out.write(image)
                    
            
            world.render(display)
            pygame.display.flip()

        if not args.no_save and not abandon_scenario:
            world.imu_sensor.toggle_recording_IMU()
            world.save_ego_data(stored_path)
            world.collision_sensor.save_history(stored_path)
            time.sleep(10)
            world.camera_manager.toggle_recording(stored_path)
            save_description(world, args, stored_path, weather, agents_dict, nearest_obstacle)
            world.finish = True

    
    # except Exception as e:
    #     print("Exception occured.")
    #     print(e)
    
    finally:
        # to save a top view video
        out.release()
        print('Closing...')
        if not args.no_save:
            client.stop_recorder() # end recording
        
        if (world and world.recording_enabled):
            client.stop_recorder()

        print('destroying vehicles')
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('destroying walkers')
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)

        if world is not None:
            world.destroy()
        
        if not args.no_save and not abandon_scenario:
            # stored_path = os.path.join(root, scenario_name)
            finish_tag = open(stored_path+'/finish.txt', 'w')
            finish_tag.close()

        pygame.quit()
    return

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--scenario_id',
        type=str,
        required=True,
        help='name of the scenario')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1536x512',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Random device seed')
    argparser.add_argument(
        '--weather',
        default='ClearNoon',
        type=str,
        # choices=['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'MidRainyNoon', 'HardRainNoon', 'SoftRainNoon',
        #          'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset', 'MidRainSunset', 'HardRainSunset', 'SoftRainSunset',
        #          'ClearNight', 'CloudyNight', 'WetNight', 'WetCloudyNight', 'MidRainyNight', 'HardRainNight', 'SoftRainNight'],
        help='weather name')
    argparser.add_argument(
        '--map',
        default='Town03',
        type=str,
        required=True,
        help='map name')
    argparser.add_argument(
        '--random_actors',
        type=str,
        default='high',
        choices=['high'],
        help='enable roaming actors')
    argparser.add_argument(
        '--scenario_type',
        type=str,
        choices=['interactive', 'collision', 'obstacle', 'non-interactive'],
        required=True,
        help='enable roaming actors')
    argparser.add_argument(
        '--no_save',
        default=False,
        action='store_true',
        help='run scenarios only')
    argparser.add_argument(
        '--save_rss',
        default=False,
        action='store_true',
        help='save rss predictinos')
    argparser.add_argument(
        '--replay',
        default=False,
        action='store_true',
        help='use random seed to generate the same behavior')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    # exec("args.weather = carla.WeatherParameters.%s" % args.weather)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
