#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

from __future__ import print_function
import glob
import os
import sys

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    sys.path.append('../carla/agents')
    sys.path.append('../carla/')
    sys.path.append('../../HDMaps')
    sys.path.append('rss/')  # rss

except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

from util.read_input import *
from util.get_and_control_trafficlight import *
from util.random_actors import spawn_actor_nearby

import carla
from carla import VehicleLightState as vls
from carla import ColorConverter as cc
# from carla import Transform, Location, Rotation
from util.controller import VehiclePIDController

import argparse
import collections
import datetime
import logging
import math
import random
import cv2
import csv
import json
import re
import weakref
import time
import threading
from multiprocessing import Process
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')

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


def write_json(filename, index, seed):
    with open(filename, 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        y = {str(index): seed}
        file_data.update(y)
        file.seek(0)
        json.dump(file_data, file, indent=4)

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, client_bp, hud, args, seeds):
        self.world = carla_world
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.abandon_scenario = False
        self.finish = False
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True  # Enables synchronous mode
        self.world.apply_settings(settings)
        self.actor_role_name = args.rolename
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
        self.restart(self.args, seeds)
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

    def restart(self, args, seeds):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.

        seed_1 = seeds[1]
        random.seed(seed_1)

        # print("seed_1: ", seed_1)
        random.seed(seed_1)

        blueprint = random.choice(
            self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)

        if blueprint.has_attribute('color'):

            seed_2 = seeds[2]
            random.seed(seed_2)

            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        if blueprint.has_attribute('driver_id'):

            seed_3 = seeds[2]
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
        self.camera_manager = CameraManager(
            self.player, self.hud, self._gamma, self.save_mode)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.camera_manager.background = True
        self.camera_manager.save_mode = self.save_mode

        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

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

                self.camera_manager.sensor_top,
                self.camera_manager.sensor_ss_top,

                self.camera_manager.sensor_rgb_front,
                self.camera_manager.sensor_ss_front,
                self.camera_manager.sensor_depth_front,

                self.camera_manager.sensor_rgb_left,
                self.camera_manager.sensor_ss_left,
                self.camera_manager.sensor_depth_left,

                self.camera_manager.sensor_rgb_right,
                self.camera_manager.sensor_ss_right,
                self.camera_manager.sensor_depth_right,

                # self.camera_manager.sensor_rgb_rear,
                # self.camera_manager.sensor_ss_rear,
                # self.camera_manager.sensor_depth_rear,
                # self.camera_manager.sensor_rgb_rear_left,
                # self.camera_manager.sensor_ss_rear_left,
                # self.camera_manager.sensor_depth_rear_left,
                # self.camera_manager.sensor_rgb_rear_right,
                # self.camera_manager.sensor_ss_rear_right,
                # self.camera_manager.sensor_depth_rear_right,

                self.camera_manager.sensor_lidar,

                self.collision_sensor.sensor,
                self.lane_invasion_sensor.sensor,
                self.gnss_sensor.sensor,
                self.imu_sensor.sensor
            ]

        else:
            sensors = [
                # self.camera_manager.sensor_lbc_img,
                self.camera_manager.sensor_top,
                self.collision_sensor.sensor,
                self.lane_invasion_sensor.sensor,
                self.gnss_sensor.sensor,
                self.imu_sensor.sensor
            ]

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

        # compass = world.imu_sensor.compass
        # heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        # heading += 'S' if 90.5 < compass < 269.5 else ''
        # heading += 'E' if 0.5 < compass < 179.5 else ''
        # heading += 'W' if 180.5 < compass < 359.5 else ''
        # colhist = world.collision_sensor.get_collision_history()
        # collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        # max_col = max(1.0, max(collision))
        # collision = [x / max_col for x in collision]

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
        self.other_actor_id = 0  # init as 0 for static object
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

    # def get_collision_history(self):
    #     history = collections.defaultdict(int)
    #     for frame, intensity in self.history:
    #         history[frame] += intensity
    #     return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        # impulse = event.normal_impulse
        # intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        # dict: {data1, data2}
        # data = frame: {timestamp, other_actor's id, intensity}
        self.history.append(
            {'frame': event.frame, 'actor_id': event.other_actor.id})
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
        self.frame = 0

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
        self.frame = sensor_data.frame

        # if self.recording:
        #     imu = {'accelerometer_x': self.accelerometer[0], 'accelerometer_y': self.accelerometer[1],
        #            'accelerometer_z': self.accelerometer[2], 'gyroscope_x': self.gyroscope[0],
        #            'gyroscope_y': self.gyroscope[1], 'gyroscope_z': self.gyroscope[2],
        #            'compass': self.compass}
        #     # self.imu_save.append([sensor_data.frame,
        #     #                     self.accelerometer[0], self.accelerometer[1], self.accelerometer[2],
        #     #                     self.gyroscope[0], self.gyroscope[1], self.gyroscope[2],
        #     #                     self.compass])
        #     if not sensor_data.frame in self.ego_dict:
        #         self.ego_dict[sensor_data.frame] = {}
        #     self.ego_dict[sensor_data.frame]['imu'] = imu
        #     self.ego_dict[sensor_data.frame]['timestamp'] = sensor_data.timestamp

    def toggle_recording_IMU(self):
        self.recording = not self.recording
    #     if not self.recording:
    #         t_top = threading.Thread(target = self.save_IMU, args=(self.imu_save, path))
    #         t_top.start()
    #         self.imu_save = []

    # def save_IMU(self, save_list, path):
    #     np_imu = np.asarray(save_list)
    #     np.save('%s/imu' % (path), np_imu)
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
    
        self.ss_top = None
        self.sensor_top = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.save_mode = save_mode

        self.rgb_front = None
        self.rgb_left = None
        self.rgb_right = None
        # self.rgb_rear = None
        # self.rgb_rear_left = None
        # self.rgb_rear_right = None

        self.ss_front = None
        self.ss_left = None
        self.ss_right = None
        # self.ss_rear = None
        # self.ss_rear_left = None
        # self.ss_rear_right = None

        self.depth_front = None
        self.depth_left = None
        self.depth_right = None
        # self.depth_rear = None
        # self.depth_rear_left = None
        # self.depth_rear_right = None

        self.lidar = None

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
                 z=100.0), carla.Rotation(pitch=-90.0)), Attachment.SpringArm),


                # sensor config for transfuser camera settings
                #  front view 8
                (carla.Transform(carla.Location(x=1.3, y=0,
                 z=2.0), carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0)), Attachment.Rigid),
                # left view  9
                (carla.Transform(carla.Location(x=1.3, y=0,
                 z=2.0), carla.Rotation(roll=0.0, pitch=0.0, yaw=-60.0)), Attachment.Rigid),
                # right view 10
                (carla.Transform(carla.Location(x=1.3, y=0,
                 z=2.0), carla.Rotation(roll=0.0, pitch=0.0, yaw=60.0)), Attachment.Rigid),
                # rear 11
                (carla.Transform(carla.Location(x=-1.3, y=0,
                 z=2.0), carla.Rotation(roll=0.0, pitch=0.0, yaw=180.0)), Attachment.Rigid),
                # rear left 12
                (carla.Transform(carla.Location(x=-1.3, y=0,
                 z=2.0), carla.Rotation(roll=0.0, pitch=0.0, yaw=-120.0)), Attachment.Rigid),
                # rear right 13
                (carla.Transform(carla.Location(x=-1.3, y=0,
                 z=2.0), carla.Rotation(roll=0.0, pitch=0.0, yaw=120.0)), Attachment.Rigid),
                # ins top 14 
                (carla.Transform(carla.Location(x=0.0, y=0.0,
                 z=50.0), carla.Rotation(pitch=-90.0)), Attachment.Rigid),

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
        # self.bev_bp.set_attribute('image_size_x', str(512))
        # self.bev_bp.set_attribute('image_size_y', str(512))
        # self.bev_bp.set_attribute('fov', str(50.0))
        # if self.bev_bp.has_attribute('gamma'):
        #     self.bev_bp.set_attribute('gamma', str(gamma_correction))

        self.bev_seg_bp = bp_library.find('sensor.camera.instance_segmentation')
        self.bev_seg_bp.set_attribute('image_size_x', str(512))
        self.bev_seg_bp.set_attribute('image_size_y', str(512))
        self.bev_seg_bp.set_attribute('fov', str(50.0))

        self.sensor_rgb_bp = bp_library.find('sensor.camera.rgb')
        self.sensor_rgb_bp.set_attribute('image_size_x', str(960))
        self.sensor_rgb_bp.set_attribute('image_size_y', str(480))
        self.sensor_rgb_bp.set_attribute('fov', str(60.0))

        self.sensor_ss_bp = bp_library.find(
            'sensor.camera.instance_segmentation')
        self.sensor_ss_bp.set_attribute('image_size_x', str(960))
        self.sensor_ss_bp.set_attribute('image_size_y', str(480))
        self.sensor_ss_bp.set_attribute('fov', str(60.0))

        self.sensor_depth_bp = bp_library.find('sensor.camera.depth')
        self.sensor_depth_bp.set_attribute('image_size_x', str(960))
        self.sensor_depth_bp.set_attribute('image_size_y', str(480))
        self.sensor_depth_bp.set_attribute('fov', str(60.0))

        self.sensor_lidar_bp = bp_library.find('sensor.lidar.ray_cast')
        self.sensor_lidar_bp.set_attribute('range', str(50))

        for item in self.sensors:

            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            # elif item[0].startswith('sensor.lidar'):
            #     self.lidar_range = 50

            #     for attr_name, attr_value in item[3].items():
            #         bp.set_attribute(attr_name, attr_value)
            #         if attr_name == 'range':
            #             self.lidar_range = float(attr_value)

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
            if self.sensor_top is not None:
                self.sensor_top.destroy()
                self.surface = None

            # rgb sensor
            # self.sensor_lbc_img = self._parent.get_world().spawn_actor(
            #     self.bev_bp,
            #     self._camera_transforms[7][0],
            #     attach_to=self._parent)

            self.sensor_top = self._parent.get_world().spawn_actor(
                self.sensors[0][-1],
                self._camera_transforms[6][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[6][1])

            if self.save_mode:
                
                # inst top 
                self.sensor_ss_top = self._parent.get_world().spawn_actor(
                    self.bev_seg_bp,
                    self._camera_transforms[14][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[14][1])
                # front

                self.sensor_rgb_front = self._parent.get_world().spawn_actor(
                    self.sensor_rgb_bp,
                    self._camera_transforms[8][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

                self.sensor_ss_front = self._parent.get_world().spawn_actor(
                    self.sensor_ss_bp,
                    self._camera_transforms[8][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

                self.sensor_depth_front = self._parent.get_world().spawn_actor(
                    self.sensor_depth_bp,
                    self._camera_transforms[8][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

                # left
                self.sensor_rgb_left = self._parent.get_world().spawn_actor(
                    self.sensor_rgb_bp,
                    self._camera_transforms[9][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

                self.sensor_ss_left = self._parent.get_world().spawn_actor(
                    self.sensor_ss_bp,
                    self._camera_transforms[9][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

                self.sensor_depth_left = self._parent.get_world().spawn_actor(
                    self.sensor_depth_bp,
                    self._camera_transforms[9][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

                self.sensor_rgb_right = self._parent.get_world().spawn_actor(
                    self.sensor_rgb_bp,
                    self._camera_transforms[10][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

                self.sensor_ss_right = self._parent.get_world().spawn_actor(
                    self.sensor_ss_bp,
                    self._camera_transforms[10][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

                self.sensor_depth_right = self._parent.get_world().spawn_actor(
                    self.sensor_depth_bp,
                    self._camera_transforms[10][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

                # # rear
                # self.sensor_rgb_rear = self._parent.get_world().spawn_actor(
                #     self.sensor_rgb_bp,
                #     self._camera_transforms[11][0],
                #     attach_to=self._parent,
                #     attachment_type=self._camera_transforms[0][1])

                # self.sensor_ss_rear = self._parent.get_world().spawn_actor(
                #     self.sensor_ss_bp,
                #     self._camera_transforms[11][0],
                #     attach_to=self._parent,
                #     attachment_type=self._camera_transforms[0][1])

                # self.sensor_depth_rear = self._parent.get_world().spawn_actor(
                #     self.sensor_depth_bp,
                #     self._camera_transforms[11][0],
                #     attach_to=self._parent,
                #     attachment_type=self._camera_transforms[0][1])

                # # rear left
                # self.sensor_rgb_rear_left = self._parent.get_world().spawn_actor(
                #     self.sensor_rgb_bp,
                #     self._camera_transforms[12][0],
                #     attach_to=self._parent,
                #     attachment_type=self._camera_transforms[0][1])

                # self.sensor_ss_rear_left = self._parent.get_world().spawn_actor(
                #     self.sensor_ss_bp,
                #     self._camera_transforms[12][0],
                #     attach_to=self._parent,
                #     attachment_type=self._camera_transforms[0][1])

                # self.sensor_depth_rear_left = self._parent.get_world().spawn_actor(
                #     self.sensor_depth_bp,
                #     self._camera_transforms[12][0],
                #     attach_to=self._parent,
                #     attachment_type=self._camera_transforms[0][1])

                # # rear left
                # self.sensor_rgb_rear_right = self._parent.get_world().spawn_actor(
                #     self.sensor_rgb_bp,
                #     self._camera_transforms[13][0],
                #     attach_to=self._parent,
                #     attachment_type=self._camera_transforms[0][1])

                # self.sensor_ss_rear_right = self._parent.get_world().spawn_actor(
                #     self.sensor_ss_bp,
                #     self._camera_transforms[13][0],
                #     attach_to=self._parent,
                #     attachment_type=self._camera_transforms[0][1])

                # self.sensor_depth_rear_right = self._parent.get_world().spawn_actor(
                #     self.sensor_depth_bp,
                #     self._camera_transforms[13][0],
                #     attach_to=self._parent,
                #     attachment_type=self._camera_transforms[0][1])

                # lidar sensor
                self.sensor_lidar = self._parent.get_world().spawn_actor(
                    # self.sensors[6][-1],
                    self.sensor_lidar_bp,
                    self._camera_transforms[0][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            # self.sensor_lbc_img.listen(
            #     lambda image: CameraManager._parse_image(weak_self, image, 'lbc_img'))
            self.sensor_top.listen(
                lambda image: CameraManager._parse_image(weak_self, image, 'top'))

            if self.save_mode:
                
                
                self.sensor_ss_top.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'ss_top'))
                
                self.sensor_rgb_front.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'rgb_front'))
                self.sensor_rgb_left.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'rgb_left'))
                self.sensor_rgb_right.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'rgb_right'))
                # self.sensor_rgb_rear.listen(lambda image: CameraManager._parse_image(weak_self, image, 'rgb_rear'))
                # self.sensor_rgb_rear_left.listen(lambda image: CameraManager._parse_image(weak_self, image, 'rgb_rear_left'))
                # self.sensor_rgb_rear_right.listen(lambda image: CameraManager._parse_image(weak_self, image, 'rgb_rear_right'))

                self.sensor_ss_front.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'ss_front'))
                self.sensor_ss_left.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'ss_left'))
                self.sensor_ss_right.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'ss_right'))
                # self.sensor_ss_rear.listen(lambda image: CameraManager._parse_image(weak_self, image, 'ss_rear'))
                # self.sensor_ss_rear_left.listen(lambda image: CameraManager._parse_image(weak_self, image, 'ss_rear_left'))
                # self.sensor_ss_rear_right.listen(lambda image: CameraManager._parse_image(weak_self, image, 'ss_rear_right'))

                self.sensor_depth_front.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'depth_front'))
                self.sensor_depth_left.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'depth_left'))
                self.sensor_depth_right.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'depth_right'))
                # self.sensor_depth_rear.listen(lambda image: CameraManager._parse_image(weak_self, image, 'depth_rear'))
                # self.sensor_depth_rear_left.listen(lambda image: CameraManager._parse_image(weak_self, image, 'depth_rear_left'))
                # self.sensor_depth_rear_right.listen(lambda image: CameraManager._parse_image(weak_self, image, 'depth_rear_right'))

                self.sensor_lidar.listen(
                    lambda image: CameraManager._parse_image(weak_self, image, 'lidar'))

                # # self.sensor_lbc_ins.listen(lambda image: CameraManager._parse_image(
                # #     weak_self, image, 'lbc_ins'))
                # # self.ss_top.listen(lambda image: CameraManager._parse_image(
                # #     weak_self, image, 'ss_top'))

        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    # def save_bbox(self, path, seg_list, width_list, height_list, fov_list):
    #     # change seg to dict, key: view, frame_num
    #     start_time = time.time()
    #     path += '/bbox'
    #     seg_dict = {}
    #     for view in self.sensor_order:
    #         seg_dict[view] = {}
    #     for i,seg_l in enumerate(seg_list):
    #         for seg_img in seg_l:
    #             seg_dict[self.sensor_order[i]][seg_img.frame] = seg_img
    #     for top_img in self.top_img:
    #         try:
    #             vehicles, cam = self.snap_dict[top_img.frame]
    #             filtered = cva.auto_annotate(
    #                 vehicles, cam[0], width_list[0], height_list[0], fov_list[0])
    #             path_temp = path + '/' +self.sensor_order[0]
    #             cva.save_output(top_img, seg_dict[self.sensor_order[0]][top_img.frame], filtered['bbox'], path_temp, filtered['vehicles'],
    #                             save_patched=False, out_format='json',threshold = 0.35)
    #         except:
    #             continue
    #         for i in range(1,7):
    #             filtered = cva.auto_annotate(
    #             vehicles, cam[i], width_list[i], height_list[i], fov_list[i])
    #             path_temp = path + '/' + self.sensor_order[i]
    #             cva.save_output(self.img_dict[self.sensor_order[i]][top_img.frame], seg_dict[self.sensor_order[i]][top_img.frame], filtered['bbox'], path_temp, filtered['vehicles'],
    #                             save_patched=False, out_format='json')
    #     end_time = time.time()
    #     print('bbox save done in %s' % (end_time-start_time))

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
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]
                    ['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(
                dvs_img.swapaxes(0, 1))

        elif view == 'top':
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            # render the view shown in monitor
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            
        elif view == 'ss_top':
            self.ss_top = image
        elif view == 'rgb_front':
            self.rgb_front = image
        elif view == 'rgb_left':
            self.rgb_left = image
        elif view == 'rgb_right':
            self.rgb_right = image
        # elif view == 'rgb_rear':
        #     self.rgb_rear = image
        # elif view == 'rgb_rear_left':
        #     self.rgb_rear_left = image
        # elif view == 'rgb_rear_right':
            self.rgb_rear_right = image
        elif view == 'depth_front':
            self.depth_front = image
        elif view == 'depth_left':
            self.depth_left = image
        elif view == 'depth_right':
            self.depth_right = image
        # elif view == 'depth_rear':
        #     self.depth_rear = image
        # elif view == 'depth_rear_left':
        #     self.depth_rear_left = image
        # elif view == 'depth_rear_right':
        #     self.depth_rear_right = image
        
        elif view == 'ss_front':
            self.ss_front = image
        elif view == 'ss_left':
            self.ss_left = image
        elif view == 'ss_right':
            self.ss_right = image
            
        # elif view == 'ss_rear':
        #     self.ss_rear = image
        # elif view == 'ss_rear_left':
        #     self.ss_rear_left = image
        # elif view == 'ss_rear_right':
        #     self.ss_rear_right = image
        elif view == 'lidar':
            self.lidar = image

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

    return blueprint

# def save_description(world, args, stored_path, weather, agents_dict, nearest_obstacle):
#     vehicles = world.world.get_actors().filter('vehicle.*')
#     peds = world.world.get_actors().filter('walker.*')
#     d = dict()
#     d['num_actor'] = len(vehicles) + len(peds)
#     d['num_vehicle'] = len(vehicles)
#     d['weather'] = str(weather)
#     # d['random_objects'] = args.random_objects
#     d['random_actors'] = args.random_actors
#     d['simulation_time'] = int(world.hud.simulation_time)
#     d['nearest_obstacle'] = nearest_obstacle

#     for key in agents_dict:
#         d[key] = agents_dict[key].id

#     with open('%s/dynamic_description.json' % (stored_path), 'w') as f:
#         json.dump(d, f, indent=4)

def write_actor_list(world, stored_path):

    def write_row(writer, actors, filter_str, class_id, min_id, max_id):
        filter_actors = actors.filter(filter_str)
        for actor in filter_actors:
            if actor.id < min_id:
                min_id = actor.id
            if actor.id > max_id:
                max_id = actor.id
            writer.writerow([actor.id, class_id, actor.type_id])
        return min_id, max_id

    filter_ = ['walker.*', 'vehicle.*', 'static.prop.streetbarrier*',
               'static.prop.trafficcone*', 'static.prop.trafficwarning*']
    id_ = [4, 10, 20, 20, 20]
    actors = world.world.get_actors()
    min_id = int(1e7)
    max_id = int(0)
    with open(stored_path+'/actor_list.csv', 'w') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(['Actor_ID', 'Class', 'Blueprint'])
        for filter_str, class_id in zip(filter_, id_):
            min_id, max_id = write_row(
                writer, actors, filter_str, class_id, min_id, max_id)
        print('min id: {}, max id: {}'.format(min_id, max_id))
    return min_id, max_id


def generate_obstacle(world, bp, src_path, ego_transform, stored_path):
    """
        stored_path : data_collection/{scenario_type}/{scenario_id}/{weather}+'_'+{random_actors}+'_'    
    """

    obstacle_list = json.load(open(src_path))
    obstacle_info = {}

    min_dis = float('Inf')
    nearest_obstacle = -1

    for obstacle_attr in obstacle_list:

        """
            obstacle_attr = {"obstacle_type": actor.type_id,
                            "basic_id": actor.id,
                            "location": new_trans.location.__dict__,
                            "rotation": new_trans.rotation.__dict__}
        """

        obstacle_name = obstacle_attr["obstacle_type"]

        if not "static" in obstacle_name:
            continue

        location = obstacle_attr["location"]
        rotation = obstacle_attr["rotation"]

        x = float(location["x"])
        y = float(location["y"])
        z = float(location["z"])
        pitch = float(rotation["pitch"])
        yaw = float(rotation["yaw"])
        roll = float(rotation["roll"])

        obstacle_loc = carla.Location(x, y, z)
        obstacle_rot = carla.Rotation(pitch, yaw, roll)
        obstacle_trans = carla.Transform(obstacle_loc, obstacle_rot)

        obstacle_actor = world.spawn_actor(
            bp.filter(obstacle_name)[0], obstacle_trans)

        dis = ego_transform.location.distance(obstacle_loc)
        if dis < min_dis:
            nearest_obstacle = obstacle_actor.id
            min_dis = dis

        obstacle_info[obstacle_actor.id] = obstacle_attr

    return nearest_obstacle, obstacle_info

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


class Data_Collection():
    def __init__(self) -> None:

        self.scenario_type = "interactive"
        self.gt_interactor = -1

        self.rgb_front = []
        self.rgb_left = []
        self.rgb_right = []
        # self.rgb_rear = []
        # self.rgb_rear_left = []
        # self.rgb_rear_right = []

        self.ss_front = []
        self.ss_left = []
        self.ss_right = []
        # self.ss_rear = []
        # self.ss_rear_left = []
        # self.ss_rear_right = []

        self.depth_front = []
        self.depth_left = []
        self.depth_right = []
        # self.depth_rear = []
        # self.depth_rear_left = []
        # self.depth_rear_right = []
        self.sensor_lidar = []
    
        self.ss_top = []

        self.frame_list = []
        self.data_list = []
        self.sensor_data_list = []
        self.ego_list = []
        self.topology_list = []

        self.static_dict = {}

        self.compass = 0

        self.actor_attri_dict = {}


    def set_attribute(self, scenario_type, scenario_id, weather, actor, random_seed, map ):
        self.scenario_type = scenario_type
        self.scenario_id = scenario_id
        self.weather = weather
        self.actor = actor 
        self.seed = random_seed
        self.map = map

    def set_start_frame(self, frame):
        self.start_frame = frame

    def set_end_frame(self, frame):
        self.end_frame = frame

    def set_scenario_type(self, sceanrio):
        self.scenario_type = sceanrio

    def set_ego_id(self, world):
        self.ego_id = world.player.id

    def set_gt_interactor(self, id):
        self.gt_interactor = id

    # collect each frame data 
    def collect_sensor(self, frame, world):
        
        while True:
            if world.camera_manager.ss_top.frame == frame:
                
                self.ss_top.append(world.camera_manager.ss_top)
                break
        while True:
            if world.camera_manager.rgb_front.frame == frame:
                self.rgb_front.append(world.camera_manager.rgb_front)
                break

        while True:
            if world.camera_manager.rgb_left.frame == frame:
                self.rgb_left.append(world.camera_manager.rgb_left)
                break
        while True:
            if world.camera_manager.rgb_right.frame == frame:
                self.rgb_right.append(world.camera_manager.rgb_right)
                break
        # ss
        while True:
            if world.camera_manager.ss_front.frame == frame:
                self.ss_front.append(world.camera_manager.ss_front)
                break
        while True:
            if world.camera_manager.ss_left.frame == frame:
                self.ss_left.append(world.camera_manager.ss_left)

                break
        while True:
            if world.camera_manager.ss_right.frame == frame:
                self.ss_right.append(world.camera_manager.ss_right)
                break

        # depth
        while True:
            if world.camera_manager.depth_front.frame == frame:
                self.depth_front.append(world.camera_manager.depth_front)
                break
        while True:
            if world.camera_manager.depth_left.frame == frame:
                self.depth_left.append(world.camera_manager.depth_left)
                break
        while True:
            if world.camera_manager.depth_right.frame == frame:
                self.depth_right.append(world.camera_manager.depth_right)
                break
        while True:
            if world.camera_manager.lidar.frame == frame:
                self.sensor_lidar.append(world.camera_manager.lidar)
                break

        while True:
            if world.imu_sensor.frame == frame:
                self.compass = world.imu_sensor.compass
                break


        # store all actor
        self.frame_list.append(frame)

        self.sensor_data_list.append(self.collect_camera_data(world))

        data = self.collect_actor_data(world)

        self.data_list.append(data)
        self.ego_list.append(data[self.ego_id])
        self.topology_list.append(self.collect_topology(world))

    def collect_actor_attr(self, world):
        # Here we get all actor attributes

        def get_xyz(method, rotation=False):

            if rotation:
                roll = method.roll
                pitch = method.pitch
                yaw = method.yaw
                return {"pitch": pitch, "yaw": yaw, "roll": roll}

            else:
                x = method.x
                y = method.y
                z = method.z
                return {"x": x, "y": y, "z": z}

        ego_id = {}
        interactor_id = {}
        interactor_id[self.gt_interactor] = {}

        _id = world.player.id
        ego_type_id = world.player.type_id
        ego_semantic_tags = world.player.semantic_tags
        ego_attributes = world.player.attributes
        ego_bbox = world.player.bounding_box
        ego_bounding_box = {"extent": get_xyz(ego_bbox.extent), "location": get_xyz(
            ego_bbox.location)}

        ego_id[_id] = {}
        ego_id[_id]["type_id"] = ego_type_id
        ego_id[_id]["semantic_tags"] = ego_semantic_tags
        ego_id[_id]["attributes"] = ego_attributes
        ego_id[_id]["bounding_box"] = ego_bounding_box

        vehicle_ids = {}
        pedestrian_ids = {}
        traffic_light_ids = {}
        obstacle_ids = {}

        vehicles = world.world.get_actors().filter("*vehicle*")
        for actor in vehicles:

            _id = actor.id

            type_id = actor.type_id
            semantic_tags = actor.semantic_tags
            attributes = actor.attributes
            bbox = actor.bounding_box
            bounding_box = {"extent": get_xyz(bbox.extent), "location": get_xyz(
                bbox.location)}
            
            vehicle_ids[_id] = {}
            vehicle_ids[_id]["type_id"] = type_id
            vehicle_ids[_id]["semantic_tags"] = semantic_tags
            vehicle_ids[_id]["attributes"] = attributes
            vehicle_ids[_id]["bounding_box"] = bounding_box

        walkers = world.world.get_actors().filter("*pedestrian*")
        for actor in walkers:

            _id = actor.id

            type_id = actor.type_id
            semantic_tags = actor.semantic_tags
            attributes = actor.attributes
            bbox = actor.bounding_box
            bounding_box = {"extent": get_xyz(bbox.extent), "location": get_xyz(
                bbox.location)}
            
            pedestrian_ids[_id] = {}
            pedestrian_ids[_id]["type_id"] = type_id
            pedestrian_ids[_id]["semantic_tags"] = semantic_tags
            pedestrian_ids[_id]["attributes"] = attributes
            pedestrian_ids[_id]["bounding_box"] = bounding_box

        lights = world.world.get_actors().filter("*traffic_light*")
        for actor in lights:

            _id = actor.id

            type_id = actor.type_id
            semantic_tags = actor.semantic_tags

            actor_loc = actor.get_location()
            location = get_xyz(actor_loc)
            rotation = get_xyz(actor.get_transform().rotation, True)

            bbox = actor.bounding_box
            bounding_box = {"extent": get_xyz(bbox.extent), "location": get_xyz(
                bbox.location)}

            cord_bounding_box = {}
            verts = [v for v in bbox.get_world_vertices(
                actor.get_transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1

            traffic_light_ids[_id] = {}
            traffic_light_ids[_id]["type_id"] = type_id
            traffic_light_ids[_id]["semantic_tags"] = semantic_tags
            traffic_light_ids[_id]["location"] = location
            traffic_light_ids[_id]["rotation"] = rotation
            traffic_light_ids[_id]["bounding_box"] = bounding_box
            traffic_light_ids[_id]["cord_bounding_box"] = cord_bounding_box

        obstacle = world.world.get_actors().filter("*static.prop*")
        for actor in obstacle:

            _id = actor.id

            type_id = actor.type_id
            semantic_tags = actor.semantic_tags
            attributes = actor.attributes

            actor_loc = actor.get_location()
            location = get_xyz(actor_loc)
            rotation = get_xyz(actor.get_transform().rotation, True)

            bbox = actor.bounding_box
            bounding_box = {"extent": get_xyz(bbox.extent), "location": get_xyz(
                bbox.location), "rotation": get_xyz(bbox.rotation, True)}

            cord_bounding_box = {}
            verts = [v for v in bbox.get_world_vertices(
                actor.get_transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1

            obstacle_ids[_id] = {}
            obstacle_ids[_id]["type_id"] = type_id
            obstacle_ids[_id]["semantic_tags"] = semantic_tags
            obstacle_ids[_id]["attributes"] = attributes
            obstacle_ids[_id]["location"] = location
            obstacle_ids[_id]["rotation"] = rotation
            obstacle_ids[_id]["bounding_box"] = bounding_box
            obstacle_ids[_id]["cord_bounding_box"] = cord_bounding_box

        self.actor_attri_dict = {"vehicle": vehicle_ids,
                        "pedestrian": pedestrian_ids, 
                        "traffic_light": traffic_light_ids, 
                        "obstacle": obstacle_ids}

    def collect_topology(self, get_world):
        town_map = get_world.world.get_map()
        try:
            while True:
                if get_world.abandon_scenario:
                    print('Abandom, killing thread.')
                    return
                waypoint = town_map.get_waypoint(
                    get_world.player.get_location())
                waypoint_list = town_map.generate_waypoints(2.0)
                nearby_waypoint = []
                roads = []
                all = []
                for wp in waypoint_list:
                    dist_x = int(wp.transform.location.x) - \
                        int(waypoint.transform.location.x)
                    dist_y = int(wp.transform.location.y) - \
                        int(waypoint.transform.location.y)
                    if abs(dist_x) <= 37.5 and abs(dist_y) <= 37.5:
                        nearby_waypoint.append(wp)
                        roads.append((wp.road_id, wp.lane_id))
                for wp in nearby_waypoint:
                    for id in roads:
                        if wp.road_id == id[0] and wp.lane_id == id[1]:
                            all.append(((wp.road_id, wp.lane_id), wp))
                            break
                all = sorted(all, key=lambda s: s[0][1])
                temp_d = {}
                d = {}
                for (i, j), wp in all:
                    if (i, j) in temp_d:
                        temp_d[(i, j)] += 1
                    else:
                        temp_d[(i, j)] = 1
                for (i, j) in temp_d:
                    if temp_d[(i, j)] != 1:
                        d[(i, j)] = temp_d[(i, j)]
                rotate_quat = np.array([[0.0, -1.0], [1.0, 0.0]])
                lane_feature_ls = []
                for i, j in d:
                    halluc_lane_1, halluc_lane_2 = np.empty(
                        (0, 3*2)), np.empty((0, 3*2))
                    center_lane = np.empty((0, 3*2))
                    is_traffic_control = False
                    is_junction = False
                    turn_direction = None
                    for k in range(len(all)-1):
                        if (i, j) == all[k][0] and (i, j) == all[k+1][0]:
                            # may change & need traffic light
                            if all[k][1].get_landmarks(50, False):
                                is_traffic_control = True
                            if all[k][1].is_junction:
                                is_junction = True
                            # -= norm center
                            before = [all[k][1].transform.location.x,
                                      all[k][1].transform.location.y]
                            after = [all[k+1][1].transform.location.x,
                                     all[k+1][1].transform.location.y]
                            # transform.rotation.yaw can not be overwritten
                            before_yaw = all[k][1].transform.rotation.yaw
                            after_yaw = all[k+1][1].transform.rotation.yaw
                            if (before_yaw < -360.0):
                                before_yaw = before_yaw + 360.0
                            if (after_yaw < -360.0):
                                after_yaw = after_yaw + 360.0
                            if (after_yaw > before_yaw):
                                turn_direction = "right"  # right
                            elif (after_yaw < before_yaw):
                                turn_direction = "left"  # left
                            distance = []
                            for t in range(len(before)):
                                distance.append(after[t] - before[t])
                            np_distance = np.array(distance)
                            norm = np.linalg.norm(np_distance)
                            e1, e2 = rotate_quat @ np_distance / norm, rotate_quat.T @ np_distance / norm
                            lane_1 = np.hstack((before + e1 * all[k][1].lane_width/2, all[k][1].transform.location.z,
                                                after + e1 * all[k][1].lane_width/2, all[k+1][1].transform.location.z))
                            lane_2 = np.hstack((before + e2 * all[k][1].lane_width/2, all[k][1].transform.location.z,
                                                after + e2 * all[k][1].lane_width/2, all[k+1][1].transform.location.z))
                            lane_c = np.hstack((before, all[k][1].transform.location.z,
                                                after, all[k+1][1].transform.location.z))
                            halluc_lane_1 = np.vstack((halluc_lane_1, lane_1))
                            halluc_lane_2 = np.vstack((halluc_lane_2, lane_2))
                            center_lane = np.vstack((center_lane, lane_c))
                    lane_feature_ls.append(
                        [halluc_lane_1, halluc_lane_2, center_lane, turn_direction, is_traffic_control, is_junction, (i, j)])
                # print("topology collection finished")
                return lane_feature_ls

        except:
            print("topology collection error.")
        pass

    def collect_static_actor_data(self, world):
        id_counter = 0
        data = {}
        
        static = world.world.get_level_bbs(carla.CityObjectLabel.Car)
        for bbox in static:
            _id = id_counter
            id_counter+=1

            cord_bounding_box = {}
            verts = [v for v in bbox.get_world_vertices(carla.Transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1
            data[_id] = {}
            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = "Car"
        data["num_of_id"] = id_counter

        static = world.world.get_level_bbs(carla.CityObjectLabel.Truck)
        for bbox in static:
            _id = id_counter
            id_counter+=1

            cord_bounding_box = {}
            verts = [v for v in bbox.get_world_vertices(carla.Transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1
            data[_id] = {}
            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = "Truck"

        
        static = world.world.get_level_bbs(carla.CityObjectLabel.Bus)
        for bbox in static:
            _id = id_counter
            id_counter+=1

            cord_bounding_box = {}
            verts = [v for v in bbox.get_world_vertices(carla.Transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1
            data[_id] = {}
            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = "Bus"

        static = world.world.get_level_bbs(carla.CityObjectLabel.Motorcycle)
        for bbox in static:
            _id = id_counter
            id_counter+=1

            cord_bounding_box = {}
            verts = [v for v in bbox.get_world_vertices(carla.Transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1
            data[_id] = {}
            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = "Motorcycle"

        static = world.world.get_level_bbs(carla.CityObjectLabel.Bicycle)
        for bbox in static:
            _id = id_counter
            id_counter+=1

            cord_bounding_box = {}
            verts = [v for v in bbox.get_world_vertices(carla.Transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1
            data[_id] = {}
            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = "Bicycle"

        data["num_of_id"] = id_counter

        self.static_dict = data
  
    def collect_actor_data(self, world):

        vehicles_id_list = []

        def get_xyz(method, rotation=False):

            if rotation:
                roll = method.roll
                pitch = method.pitch
                yaw = method.yaw
                return {"pitch": pitch, "yaw": yaw, "roll": roll}

            else:
                x = method.x
                y = method.y
                z = method.z

                # return x, y, z
                return {"x": x, "y": y, "z": z}

        ego_loc = world.player.get_location()
        data = {}
        vehicles = world.world.get_actors().filter("*vehicle*")
        for actor in vehicles:

            _id = actor.id
            actor_loc = actor.get_location()
            location = get_xyz(actor_loc)
            rotation = get_xyz(actor.get_transform().rotation, True)

            cord_bounding_box = {}
            bbox = actor.bounding_box
            verts = [v for v in bbox.get_world_vertices(
                actor.get_transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1

            distance = ego_loc.distance(actor_loc)

            if distance < 50:
                vehicles_id_list.append(_id)

            acceleration = get_xyz(actor.get_acceleration())
            velocity = get_xyz(actor.get_velocity())
            angular_velocity = get_xyz(actor.get_angular_velocity())


            v = actor.get_velocity()


            speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)

            vehicle_control = actor.get_control()
            control = {
                "throttle": vehicle_control.throttle,
                "steer": vehicle_control.steer,
                "brake": vehicle_control.brake,
                "hand_brake": vehicle_control.hand_brake,
                "reverse": vehicle_control.reverse,
                "manual_gear_shift": vehicle_control.manual_gear_shift,
                "gear": vehicle_control.gear
            }

            data[_id] = {}
            data[_id]["location"] = location
            data[_id]["rotation"] = rotation
            data[_id]["distance"] = distance
            data[_id]["acceleration"] = acceleration
            data[_id]["velocity"] = velocity
            data[_id]["speed"] = speed
            data[_id]["angular_velocity"] = angular_velocity
            data[_id]["control"] = control
            if _id == self.ego_id:
                data[_id]["compass"] = self.compass
            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = "vehicle"
            

        pedestrian_id_list = []

        walkers = world.world.get_actors().filter("*pedestrian*")
        for actor in walkers:

            _id = actor.id

            actor_loc = actor.get_location()
            location = get_xyz(actor_loc)
            rotation = get_xyz(actor.get_transform().rotation, True)

            cord_bounding_box = {}
            bbox = actor.bounding_box
            verts = [v for v in bbox.get_world_vertices(
                actor.get_transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1

            distance = ego_loc.distance(actor_loc)

            if distance < 50:
                pedestrian_id_list.append(_id)

            acceleration = get_xyz(actor.get_acceleration())
            velocity = get_xyz(actor.get_velocity())
            angular_velocity = get_xyz(actor.get_angular_velocity())

            walker_control = actor.get_control()
            control = {"direction": get_xyz(walker_control.direction),
                       "speed": walker_control.speed, "jump": walker_control.jump}

            data[_id] = {}
            data[_id]["location"] = location
            # data[_id]["rotation"] = rotation
            data[_id]["distance"] = distance
            data[_id]["acceleration"] = acceleration
            data[_id]["velocity"] = velocity
            data[_id]["angular_velocity"] = angular_velocity
            data[_id]["control"] = control

            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = 'pedestrian'

        traffic_id_list = []

        lights = world.world.get_actors().filter("*traffic_light*")
        for actor in lights:

            _id = actor.id

            traffic_light_state = int(actor.state)  # traffic light state
            distance = ego_loc.distance(actor_loc)

            if distance < 50:
                traffic_id_list.append(_id)

            data[_id] = {}
            data[_id]["state"] = traffic_light_state
            actor_loc = actor.get_location()
            location = get_xyz(actor_loc)
            data[_id]["location"] = location
            data[_id]["distance"] = distance
            data[_id]["type"] = "traffic_light"

            trigger = actor.trigger_volume
                        # bbox = actor.bounding_box
            verts = [v for v in trigger.get_world_vertices(carla.Transform())]

            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1
            data[_id]["tigger_cord_bounding_box"] = cord_bounding_box
            box = trigger.extent
            loc = trigger.location
            ori = trigger.rotation.get_forward_vector()
            data[_id]["trigger_loc"] = [loc.x, loc.y, loc.z]
            data[_id]["trigger_ori"] = [ori.x, ori.y, ori.z]
            data[_id]["trigger_box"] = [box.x, box.y]

        obstacle_id_list = []

        obstacle = world.world.get_actors().filter("*static.prop*")
        for actor in obstacle:

            _id = actor.id

            distance = ego_loc.distance(actor_loc)

            if distance < 50:
                obstacle_id_list.append(_id)

            data[_id] = {}
            data[_id]["distance"] = distance
            data[_id]["type"] = "obstacle"


        data["ego_id"] = world.player.id
        if self.scenario_type != "non-interactive":
            data["interactor_id"] = self.gt_interactor
        else:
            data["interactor_id"] = -1

        data["obstacle_ids"] = obstacle_id_list
        data["traffic_light_ids"] = traffic_id_list
        data["vehicles_ids"] = vehicles_id_list
        data["pedestrian_ids"] = pedestrian_id_list
            
    
        return data

    def _get_forward_speed(self, transform, velocity):
        """ Convert the vehicle transform directly to forward speed """

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array(
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def collect_camera_data(self, world):

        data = {}

        intrinsic = np.identity(3)
        intrinsic[0, 2] = 960 / 2.0
        intrinsic[1, 2] = 480 / 2.0
        intrinsic[0, 0] = intrinsic[1, 1] = 960 / (
            2.0 * np.tan(60 * np.pi / 360.0)
        )
        # sensor_location
        data["front"] = {}
        data["front"]["extrinsic"] = world.camera_manager.sensor_rgb_front.get_transform(
        ).get_matrix()  # camera 2 world
        data["front"]["intrinsic"] = intrinsic
        sensor = world.camera_manager.sensor_rgb_front
        data["front"]["loc"] = np.array(
            [sensor.get_location().x, sensor.get_location().y, sensor.get_location().z])
        data["front"]["w2c"] = np.array(
            world.camera_manager.sensor_rgb_front.get_transform().get_inverse_matrix())

        data["left"] = {}
        data["left"]["extrinsic"] = world.camera_manager.sensor_rgb_left.get_transform(
        ).get_matrix()
        data["left"]["intrinsic"] = intrinsic
        sensor = world.camera_manager.sensor_rgb_left
        data["left"]["loc"] = np.array(
            [sensor.get_location().x, sensor.get_location().y, sensor.get_location().z])
        data["left"]["w2c"] = np.array(
            world.camera_manager.sensor_rgb_left.get_transform().get_inverse_matrix())

        data["right"] = {}
        data["right"]["extrinsic"] = world.camera_manager.sensor_rgb_right.get_transform(
        ).get_matrix()
        data["right"]["intrinsic"] = intrinsic
        sensor = world.camera_manager.sensor_rgb_right
        data["right"]["loc"] = np.array(
            [sensor.get_location().x, sensor.get_location().y, sensor.get_location().z])
        data["right"]["w2c"] = np.array(
            world.camera_manager.sensor_rgb_right.get_transform().get_inverse_matrix())
        
        intrinsic = np.identity(3)
        intrinsic[0, 2] = 512 / 2.0
        intrinsic[1, 2] = 512 / 2.0
        intrinsic[0, 0] = intrinsic[1, 1] = 512 / (
            2.0 * np.tan(50 * np.pi / 360.0)
        )
        
        data["top"] = {}
        data["top"]["extrinsic"] = world.camera_manager.sensor_ss_top.get_transform(
        ).get_matrix()
        data["right"]["intrinsic"] = intrinsic
        sensor = world.camera_manager.sensor_ss_top
        data["right"]["loc"] = np.array(
            [sensor.get_location().x, sensor.get_location().y, sensor.get_location().z])
        data["right"]["w2c"] = np.array(
            world.camera_manager.sensor_ss_top.get_transform().get_inverse_matrix())

        # data["rear"] = {}
        # data["rear"]["extrinsic"] = world.camera_manager.sensor_rgb_rear.get_transform().get_matrix()
        # data["rear"]["intrinsic"] = intrinsic
        # sensor = world.camera_manager.sensor_rgb_rear
        # data["rear"]["loc"] = np.array([sensor.get_location().x,sensor.get_location().y,sensor.get_location().z])
        # data["rear"]["w2c"] = np.array(world.camera_manager.sensor_rgb_rear.get_transform().get_inverse_matrix())

        # data["rear_left"] = {}
        # data["rear_left"]["extrinsic"] = world.camera_manager.sensor_rgb_rear_left.get_transform().get_matrix()
        # data["rear_left"]["intrinsic"] = intrinsic
        # sensor = world.camera_manager.sensor_rgb_rear_left
        # data["rear_left"]["loc"] = np.array([sensor.get_location().x,sensor.get_location().y,sensor.get_location().z])
        # data["rear_left"]["w2c"] = np.array(world.camera_manager.sensor_rgb_rear_left.get_transform().get_inverse_matrix())

        # data["rear_right"] = {}
        # data["rear_right"]["extrinsic"] = world.camera_manager.sensor_rgb_rear_right.get_transform().get_matrix()
        # data["rear_right"]["intrinsic"] = intrinsic
        # sensor = world.camera_manager.sensor_rgb_rear_right
        # data["rear_right"]["loc"] = np.array([sensor.get_location().x,sensor.get_location().y,sensor.get_location().z])
        # data["rear_right"]["w2c"] = np.array(world.camera_manager.sensor_rgb_rear_right.get_transform().get_inverse_matrix())

        return data

    def save_json_data(self, frame_list, data_list, path, start_frame, end_frame, folder_name):

        counter = 0
        stored_path = os.path.join(path, folder_name)
        if not os.path.exists(stored_path):
            os.makedirs(stored_path)
        for idx in range(len(frame_list)):
            frame = frame_list[idx]
            data = data_list[idx]
            if (frame >= start_frame) and (frame < end_frame):
                frame = frame - start_frame
                counter += 1
                actors_data_file = stored_path + ("/%08d.json" % frame)
                f = open(actors_data_file, "w")
                json.dump(data, f, indent=4)
                f.close()

        print(folder_name + " save finished. Total: ", counter)

    def save_np_data(self, frame_list, data_list, path, start_frame, end_frame, folder_name):

        counter = 0

        stored_path = os.path.join(path, folder_name)
        if not os.path.exists(stored_path):
            os.makedirs(stored_path)
        for idx in range(len(frame_list)):

            frame = frame_list[idx]
            data = data_list[idx]
            if (frame >= start_frame) and (frame < end_frame):
                frame = frame-start_frame
                
                counter += 1
                sensor_data_file = stored_path + ("/%08d.npy" % frame)
                np.save(sensor_data_file, np.array(data, dtype=object))

        print(folder_name + " save finished. Total: ", counter)

    def save_img(self, img_list, sensor, path, start_frame, end_frame, view='top'):

        sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
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

        modality = sensors[sensor][0].split('.')[-1]
        counter = 0
        for img in img_list:
            if (img.frame >= start_frame) and (img.frame < end_frame):

                counter += 1
                frame = img.frame - start_frame

                if 'seg' in modality:
                    img.save_to_disk(
                        '%s/%s/%s/%08d' % (path, modality, view, frame), cc.CityScapesPalette)
                elif 'depth' in modality:
                    img.save_to_disk(
                        '%s/%s/%s/%08d' % (path, modality, view, frame) , cc.LogarithmicDepth)
                elif 'lidar' in view:
                    points = np.frombuffer(img.raw_data, dtype=np.dtype('f4'))
                    points = np.reshape(points, (int(points.shape[0] / 4), 4))
                    if not os.path.exists('%s/%s/' % (path, view)):
                        os.makedirs('%s/%s/' % (path, view))
                    np.save('%s/%s/%08d.npy' %
                            (path, view, frame), points, allow_pickle=True)
                else:
                    # save rgb 
                    # rgb 
                    # img.save_to_disk('%s/%s/%s/%08d' %
                    #                  (path, modality, view, frame))

                    img.convert(cc.Raw)
                    array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (img.height, img.width, 4))
                    array = array[:, :, :3]
                    # array = array[:, :, ::-1]
                    if not os.path.exists('%s/%s/%s/' % (path, modality, view)):
                        os.makedirs('%s/%s/%s/' % (path, modality, view))
                    cv2.imwrite('%s/%s/%s/%08d.jpg' % (path, modality, view, frame), array, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    # cv2.imwrite('%s/%s/%s/%08d.png' % (path, modality, view, frame), array, [cv2.IMWRITE_PNG_COMPRESSION, 7])
                    




        print("%s %s save finished. Total: %d" %
              (sensors[sensor][2], view, counter))

    def save_data(self, path):
        
        t_ss_top = Process(target=self.save_img, args=(self.ss_top, 10, path,
                              self.start_frame, self.end_frame, 'top'))

        t_rgb_front = Process(target=self.save_img, args=(self.rgb_front, 0, path,
                              self.start_frame, self.end_frame, 'front'))
        t_rgb_left = Process(target=self.save_img, args=(
            self.rgb_left, 0, path, self.start_frame, self.end_frame, 'left'))
        t_rgb_right = Process(target=self.save_img, args=(self.rgb_right, 0, path,
                              self.start_frame, self.end_frame, 'right'))
        t_ss_front = Process(target=self.save_img, args=(
            self.ss_front, 10, path, self.start_frame, self.end_frame, 'front'))
        t_ss_left = Process(target=self.save_img, args=(
            self.ss_left, 10, path, self.start_frame, self.end_frame, 'left'))
        t_ss_right = Process(target=self.save_img, args=(
            self.ss_right, 10, path, self.start_frame, self.end_frame, 'right'))
        t_depth_front = Process(target=self.save_img, args=(self.depth_front, 1, path,
                                self.start_frame, self.end_frame, 'front'))
        t_depth_left = Process(target=self.save_img, args=(self.depth_left, 1, path,
                               self.start_frame, self.end_frame, 'left'))
        t_depth_right = Process(target=self.save_img, args=(self.depth_right, 1, path,
                                self.start_frame, self.end_frame, 'right'))
        t_lidar = Process(target=self.save_img, args=(self.sensor_lidar, 6,
                          path, self.start_frame, self.end_frame, 'lidar'))
        t_actors_data = Process(target=self.save_json_data, args=(
            self.frame_list, self.data_list, path, self.start_frame, self.end_frame, "actors_data"))
        t_sensor_data = Process(target=self.save_np_data, args=(
            self.frame_list, self.sensor_data_list, path, self.start_frame, self.end_frame, "sensor_data"))
        t_ego_data = Process(target=self.save_json_data, args=(
            self.frame_list, self.ego_list, path, self.start_frame, self.end_frame, "ego_data"))

        t_topology = Process(target=self.save_np_data, args=(
            self.frame_list, self.topology_list, path, self.start_frame, self.end_frame, "topology"))

        start_time = time.time()

        t_ss_top.start()
        t_rgb_front.start()
        t_rgb_left.start()
        t_rgb_right.start()
        t_ss_front.start()
        t_ss_left.start()
        t_ss_right.start()
        t_depth_front.start()
        t_depth_left.start()
        t_depth_right.start()
        t_lidar.start()
        t_actors_data.start()
        t_sensor_data.start()
        t_ego_data.start()
        t_topology.start()

        # ------------------------------ #
        t_ss_top.join()
        t_rgb_front.join()
        t_rgb_left.join()
        t_rgb_right.join()
        t_ss_front.join()
        t_ss_left.join()
        t_ss_right.join()
        t_depth_front.join()
        t_depth_left.join()
        t_depth_right.join()
        t_lidar.join()
        t_actors_data.join()
        t_sensor_data.join()
        t_ego_data.join()
        t_topology.join()

        with open(f"{path}/static_data.json", "w") as f:
            json.dump(self.static_dict, f, indent=4)
            f.close()
            print("static_data save finished.")


        with open(f"{path}/actor_attribute.json", "w") as f:
            json.dump(self.actor_attri_dict, f, indent=4)
            f.close()
            print("actor attribute save finished.")

        with open("./result.txt", "a") as f:
            f.write(f"{self.scenario_type}#{self.scenario_id}#{self.map}#{self.weather}#{self.actor}#{self.seed}")
        
        end_time = time.time()

        print('ALL save done in %s ' % (end_time-start_time))
        print("")

        # empty list
        self.ss_top = []
        self.rgb_front = []
        self.rgb_left = []
        self.rgb_right = []
        self.ss_front = []
        self.ss_left = []
        self.ss_right = []
        self.depth_front = []
        self.depth_left = []
        self.depth_right = []
        self.ego_list = []
        self.sensor_lidar = []
        self.data_list = []
        self.sensor_data_list = []
        self.actor_attri_dict = {}
        self.frame_list = []
        self.static_dict= {}
        self.topology_list = []

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    path = os.path.join('data_collection',
                        args.scenario_type, args.scenario_id)

    

    filter_dict = {}
    try:
        for root, _, files in os.walk(path + '/filter/'):
            for name in files:
                f = open(path + '/filter/' + name, 'r')
                bp = f.readlines()[0]
                name = name.strip('.txt')
                f.close()
                filter_dict[name] = bp
        # print(filter_dict)
    except:
        print("檔案夾不存在。")

    # generate random seed
    # during replay, we use seed to generate the same behavior
    random_seed_int = int(args.random_seed)
    if random_seed_int == 0:
        random_seed_int = random.randint(0, 100000)
    random.seed(random_seed_int)
    seeds = []
    for _ in range(12):
        seeds.append(random.randint(1565169134, 2665169134))

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
    abandon_scenario = False
    scenario_name = None

    vehicles_list = []
    all_id = []
    
    collection_flag = False
    detect_start = True
    detect_end = False
    # read start position and end position 
    

    if not args.test:
        # start_end_point.json 
        with open(f"{path}/start_end_point.json") as f:
            data = json.load(f)

            start_position_x  = float(data["start_x"])
            start_position_y  = float(data["start_y"])
            end_position_x  = float(data["end_x"])
            end_position_y  = float(data["end_y"])
           
    


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
    stored_path = os.path.join('data_collection', args.scenario_type, args.scenario_id,
                               'variant_scenario', weather + "_" + args.random_actors + "_")
    

    

    if args.test:
        out = cv2.VideoWriter(f'data_collection/{args.scenario_type}/{args.scenario_id}/{args.scenario_id}.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), 20,  (640, 360))
    else:
        if not os.path.exists(stored_path):
            os.makedirs(stored_path)
        out = cv2.VideoWriter(stored_path+"/"+str(args.scenario_id)+".mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'), 20,  (640, 360))


    # pass seeds to the world
    world = World(client.load_world(args.map),
                  filter_dict['player'], hud, args, seeds)

    client.get_world().set_weather(args.weather)
    # client.get_world().set_weather(getattr(carla.WeatherParameters, args.weather))
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
        nearest_obstacle, obstacle_info = generate_obstacle(client.get_world(), blueprint_library,
                                                            path+"/obstacle/obstacle_list.json", ego_transform, stored_path)

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
                    break
                except Exception:
                    transform_spawn.location.z += 1.5

            if args.scenario_type == 'obstacle':

                dis = ego_transform.location.distance(transform_spawn.location)
                if dis < min_dis:
                    nearest_obstacle = agents_dict[actor_id].id
                    min_dis = dis

                obstacle_info[agents_dict[actor_id].id] = {
                    "obstacle_type": bp,
                    "basic_id": -1,
                    "location": {
                        "x": transform_spawn.location.x,
                        "y": transform_spawn.location.y,
                        "z": transform_spawn.location.z
                    },
                    "rotation": {
                        "pitch": transform_spawn.rotation.pitch,
                        "yaw": transform_spawn.rotation.yaw,
                        "roll": transform_spawn.rotation.roll
                    }
                }

            # set other actor id for checking collision object's identity
            world.collision_sensor.other_actor_id = agents_dict[actor_id].id

        if 'vehicle' in bp:
            controller_dict[actor_id] = VehiclePIDController(agents_dict[actor_id], args_lateral={'K_P': 1, 'K_D': 0.0, 'K_I': 0}, args_longitudinal={'K_P': 1, 'K_D': 0.0, 'K_I': 0.0},
                                                             max_throttle=1.0, max_brake=1.0, max_steering=1.0)
            try:
                agents_dict[actor_id].set_light_state(
                    carla.VehicleLightState.LowBeam)
            except:
                print('vehicle has no low beam light')

        actor_transform_index[actor_id] = 1
        finish[actor_id] = False

    if args.scenario_type == "obstacle" and not args.no_save and not args.test:
        with open(os.path.join(stored_path, "obstacle_info.json"), "w")as f:
            json.dump(obstacle_info, f, indent=4)

    # root = os.path.join('data_collection', args.scenario_type, args.scenario_id)
    scenario_name = str(weather) + '_'

    if args.random_actors != 'none':
        if args.random_actors == 'pedestrian':  # only pedestrian
            vehicles_list, all_actors, all_id = spawn_actor_nearby(args, world.world, client, seeds,  distance=30, v_ratio=0.0,
                                                                   pedestrian=40, transform_dict=transform_dict)
        elif args.random_actors == 'low':
            vehicles_list, all_actors, all_id = spawn_actor_nearby(args, world.world, client, seeds,  distance=100, v_ratio=0.3,
                                                                   pedestrian=20, transform_dict=transform_dict)
        elif args.random_actors == 'mid':
            vehicles_list, all_actors, all_id = spawn_actor_nearby(args, world.world, client, seeds,  distance=100, v_ratio=0.6,
                                                                   pedestrian=45, transform_dict=transform_dict)
        elif args.random_actors == 'high':
            vehicles_list, all_actors, all_id = spawn_actor_nearby(args, world.world, client, seeds,  distance=100, v_ratio=0.8,
                                                                   pedestrian=70, transform_dict=transform_dict)


        
    scenario_name = scenario_name + args.random_actors + '_'

    # write actor list
    # min_id, max_id = write_actor_list(world, stored_path)
    # if max_id-min_id >= 65535:
    #     print('Actor id error. Abandom.')
    #     abandon_scenario = True
    #     raise

    iter_tick = 0
    iter_start = 25
    iter_toggle = 50

    if not args.no_save:
        data_collection = Data_Collection()
        data_collection.set_scenario_type(args.scenario_type)
        data_collection.set_ego_id(world)
        data_collection.set_attribute(args.scenario_type, args.scenario_id, weather, args.random_actors, args.random_seed, args.map)

        # save for only one time 
        data_collection.collect_actor_attr(world)
        data_collection.collect_static_actor_data(world)

        
        
        
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
            


            if not args.no_save:
                if args.scenario_type == "interactive" or args.scenario_type == "collision":
                    keys = list(agents_dict.keys())
                    keys.remove('player')
                    gt_interactor_id = int(agents_dict[keys[0]].id)
                    data_collection.set_gt_interactor(gt_interactor_id)
                    

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

                        target_speed = (
                            velocity_dict[actor_id][actor_transform_index[actor_id]])*3.6
                        waypoint = transform_dict[actor_id][actor_transform_index[actor_id]]

                        agents_dict[actor_id].apply_control(
                            controller_dict[actor_id].run_step(target_speed, waypoint))
                        # agents_dict[actor_id].apply_control(controller_dict[actor_id].run_step(
                        #     (velocity_dict[actor_id][actor_transform_index[actor_id]])*3.6, transform_dict[actor_id][actor_transform_index[actor_id]]))

                        v = agents_dict[actor_id].get_velocity()
                        v = ((v.x)**2 + (v.y)**2+(v.z)**2)**(0.5)

                        # to avoid the actor slowing down for the dense location around
                        if agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) < 2.0:
                            actor_transform_index[actor_id] += 2
                        elif agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) > 6.0:
                            actor_transform_index[actor_id] += 6
                        else:
                            actor_transform_index[actor_id] += 1

                    elif 'pedestrian' in filter_dict[actor_id]:
                        # print("ped")
                        agents_dict[actor_id].apply_control(
                            ped_control_dict[actor_id][actor_transform_index[actor_id]])
                        actor_transform_index[actor_id] += 1
                else:
                    finish[actor_id] = True

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

            # if iter_tick == iter_toggle:
            #     if not args.no_save:
            #         

            elif iter_tick > iter_toggle:

                if not args.no_save and (not abandon_scenario) and collection_flag and detect_start == False:
                    # collect data in sensor's list
                    data_collection.collect_sensor(frame, world)
                    

                view = pygame.surfarray.array3d(display)
                #  convert from (width, height, channel) to (height, width, channel)
                view = view.transpose([1, 0, 2])
                #  convert from rgb to bgr
                image = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                out.write(image)
                
                ego_loc = world.player.get_location()
                x = ego_loc.x
                y = ego_loc.y
                
                # check start position
                # print("current position ", x, y)

                if not args.test:
                    if detect_start:
                        distacne = math.sqrt((x - start_position_x)**2 + (y - start_position_y)**2)
                        # print("start position", start_position_x, start_position_y)
                        # print("start ", distacne)
                        # print(" ")
                        if distacne < 1.0:
                            detect_start = False
                            collection_flag = True
                            detect_end = True
                            data_collection.set_start_frame(frame)
                            
                            
                    # check end point
                        
                    if detect_end:
                        distacne = math.sqrt((x - end_position_x)**2 + (y - end_position_y)**2)
                        # print("end ", distacne)
                        if distacne < 1.0:
                            collection_flag = False
                    
                
                if detect_end and not collection_flag:
                    print('stop scenario ')
                    break
           
            
            
            # cehck end position

        world.tick(clock)
        world.render(display)
        pygame.display.flip()
    
    if args.no_save and args.generate_random_seed and (not abandon_scenario) and not args.test :
        # save random_seed 
        with open(f'{stored_path}/seed.txt', 'w') as f:
            f.write(str(random_seed_int))
        

    if not args.no_save and not abandon_scenario and not args.test:
        data_collection.set_end_frame(frame)
        data_collection.save_data(stored_path)

    # to save a top view video
    out.release()
    print('Closing...')

    print('destroying vehicles')
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    # stop walker controllers (list is [controller, actor, controller, actor ...])
    for i in range(0, len(all_id), 2):
        all_actors[i].stop()

    print('destroying walkers')
    client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

    if world is not None:
        world.destroy()

    # if not args.no_save and not abandon_scenario:
    #     stored_path = os.path.join(root, scenario_name)
    #     finish_tag = open(stored_path+'/finish.txt', 'w')
    #     finish_tag.close()

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
        default='640x360',
        help='window resolution (default: 640x360)')
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
        '--weather',
        default='ClearNoon',
        type=str,
        choices=['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'MidRainyNoon', 'HardRainNoon', 'SoftRainNoon',
                 'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset', 'MidRainSunset', 'HardRainSunset', 'SoftRainSunset',
                 'ClearNight', 'CloudyNight', 'WetNight', 'WetCloudyNight', 'MidRainyNight', 'HardRainNight', 'SoftRainNight'],
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
        default='none',
        choices=['none', 'pedestrian', 'low', 'mid', 'high'],
        help='enable roaming actors')

    argparser.add_argument(
        '--scenario_type',
        type=str,
        choices=['interactive', 'collision', 'obstacle', 'non-interactive'],
        required=True,
        help='enable roaming actors')

    argparser.add_argument(
        '--test',
        action='store_true',
        help='test the Scenario')

    # no_save flag ( Only use few camera sensor)
    argparser.add_argument(
        '--no_save',
        # default=False,
        action='store_true',
        help='run scenarios only')
    
    argparser.add_argument(
        '--generate_random_seed',
        # default=False,
        action='store_true',
        help='run scenarios only')

    argparser.add_argument(
        '--random_seed',
        default=0,
        type=int,
        help='use random_seed to replay the same behavior ')



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
