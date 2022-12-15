#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk
    O            : set coordinate

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
from multiprocessing import Process
import json
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import time
import logging
import math
import random
import re
import weakref
import threading
from auto_random_actors import spawn_actor_nearby
import cv2
from auto_light import *

import datetime


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
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True  # Enables synchronous mode
        self.world.apply_settings(settings)
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
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

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
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
            # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            # min_d=500.0
            # for p in spawn_points:
            #     if p.location.distance(carla.Location(258.7, -248.2, 2.0)) < min_d:
            #         spawn_point= p
            spawn_points = []
            for id, coor in self.hud.intersection_coordinators.items():
                spawn_points.append(carla.Transform(carla.Location(coor.x, coor.y, 2.0), carla.Rotation(0,0,0)))
            spawn_point = random.choice(spawn_points)
            print(spawn_point)
            # spawn_point = carla.Transform(carla.Location(258.7, -248.2, 2.0), carla.Rotation(0,0,0))
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            #coor=carla.Location(location[0],location[1],location[2]+2.0)
            #self.player.set_location(coor)
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):

        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        # print(preset[1])
        while ('Night' in preset[1]):
            self._weather_index += -1 if reverse else 1
            self._weather_index %= len(self._weather_presets)
            preset = self._weather_presets[self._weather_index]
            # print(preset[1])
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
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock, frame, display, stored_path):
        self.hud.tick(self, clock, self.camera_manager, frame, display, stored_path)

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
        sensors = [
            self.camera_manager.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
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
            self._control = carla.VehicleControl()
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
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
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
                elif event.key==K_o:
                    xyz=[float(s) for s in input('Enter coordinate: x , y , z  : ').split()]
                    new_location=carla.Location(xyz[0],xyz[1],xyz[2])
                    world.player.set_location(new_location)
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
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
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
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

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
            world.player.apply_control(self._control)

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
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
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
    def __init__(self, width, height, distance=25.0, town='Town05', stored_path='', v_id=1):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.recording = False
        self.recording_time = 0
        self.stored_path = stored_path
        self.v_id = int(v_id)
        self.ego_data = {}

        self.d_2_intersection = distance
        self.d_last = distance
        # self.v_last = 0
        if town == "Town01":
            self.intersection_coordinators = {
                                            't1':
                                            (336, 327, 0.0),
                                            't2':
                                            (336, 2, 0),
                                            't3':
                                            (336, 197, 0.0),
                                            't4':
                                            (336, 132, 0.0),
                                            't5':
                                            (336, 58, 0.0),
                                            't8':
                                            (156, 56, 0.0),
                                            't9':
                                            (157, 1, 0.0),
                                            't10':
                                            (91, 195.8, 0.0),
                                            't11':
                                            (90.6, 56, 0.0),
                                            't12':
                                            (90.6, 328, 0.0), 
                                            't13':
                                            (90.6, 2, 0.0),
                                            't14':
                                            (91.6, 130.6, 0.0)}
        elif town == "Town02":
            self.intersection_coordinators = {
                                            't1':
                                            (-3, 190, 0),
                                            't2':
                                            (44, 189, 0),
                                            't3':
                                            (45, 248, 0.0),
                                            't4':
                                            (43, 303, 0.0),
                                            't5':
                                            (134, 190, 0.0),
                                            't6':
                                            (134, 238, 0.0),
                                            't7':
                                            (191, 191, 0.0),
                                            't8':
                                            (191, 237, 0.0)}

        elif town == 'Town03':
            self.intersection_coordinators = {
                                            'i1': (-80, 131, 0),
                                            'i2': (-2, 132, 0),
                                            'i4': (-78,-138, 0),
                                            'i5': (3, -138, 0),

                                            't2': (169.0, 64, 0),
                                            't3': (232, 60, 0),
                                            't6': (225, 2, 0),


                                            't7': (9.8, -200,0, 0),
                                            't8': (83, -200, 0),
                                            't10': (82, -74, 0),
                                            't11': (151, -71, 0)
                                            }

        elif town == "Town04":
            self.intersection_coordinators = {
                                            'i1':
                                            (258.7, -248.2, 0.0),
                                            'i2':
                                            (313.7, -247.2, 0.0),
                                            'i3':
                                            (310.3, -170.5, 0.0),
                                            'i4':
                                            (349.5, -170.5, 0.0),
                                            'i5':
                                            (255.0, -170.2, 0.0),
                                            'i6':
                                            (313.0, -119.7, 0.0),
                                            'i7':
                                            (200.3, -173.6, 0.0),
                                            'i8':
                                            (293.4, -247.7, 0.0),
                                            'i9':
                                            (203.8, -309.0, 0.0),
                                            't1':
                                            (256.3, -309.0, 0.0),
                                            't2':
                                            (130.5, -172.7, 0.0),
                                            't3':
                                            (61.9, -172.3, 0.0),
                                            't4':
                                            (21.2, -172.2, 0.0),
                                            't5':
                                            (256.5, -122.2, 0.0)
                                            }   


        elif town == "Town05":
            self.intersection_coordinators = {
                                            'i1':
                                            (-189.3, -89.3, 0.0), 
                                            'i2':
                                            (-190.5, 1.3, 0.0), 
                                            'i3':
                                            (-189.7, 89.6, 0.0), 
                                            'i6':
                                            (-125.6, 90.4, -0.0), 
                                            'i5':
                                            (-126.2, 0.3, -0.0), 
                                            'i14':
                                            (-126, -89.1, -0.0),
                                            'i7':
                                            (-50.5, -89.4, -0.0), 
                                            'i8':
                                            (-49.1, 0.8, 0.0), 
                                            'i9':
                                            (-50.3, 89.6, 0.0),
                                            'i12':
                                            (29.9, 90.0, 0.0), 
                                            'i11':
                                            (29.9, 0.2, 0.0), 
                                            'i10':
                                            (31.7, -89.2, 0.0),
                                            'i13':
                                            (101.9, -0.1, 0.0)}
        elif town == "Town07":
            self.intersection_coordinators = {
                                            'i1':
                                            (-103, 53, 0.0), 
                                            'i2':
                                            (-3, -3, 0), 
                                            'i3':
                                            (-150.7, -35, 0.0),
                                            't1':
                                            (-152, 50, 0),
                                            't2':
                                            (66, 60, 0),
                                            't3':
                                            (-100, -0.5, 0),
                                            't4':
                                            (-200, -35, 0), 
                                            't5':
                                            (-3, -160, 0),
                                            't6':
                                            (-100, -96, 0),
                                            't7':
                                            (-100, -62, 0),
                                            't8':
                                            (-100, -34, 0),
                                            't9':
                                            (-199, -161, 0),
                                            't10':
                                            (-2, -240, 0),
                                            't11':
                                            (-5, 60, 0),
                                            't12':
                                            (66, -1, 0),
                                            't13':
                                            (-200, 49, 0),
                                            't14':
                                            (-110, 114, 0)
                                            }

        
        elif town ==  "Town10HD":
            self.intersection_coordinators = {'i1': (-46.0, 18.8, 0.0),
                                                 't2':
                                                (42.5, 63.3, 0.0), 
                                                't3':
                                                (42.4, 29.1, 0.0), 
                                                't1':
                                                (102.3, 21.4, 0.0), 
                                                't5':
                                                (-42.0, 69.0, 0.0), 
                                                't4':
                                                (-47.0, 127.9, 0.0), 
                                                't7':
                                                (-106.2, 21.8, 0.0),
                                                't6':
                                                (-47.0, -60.2, 0.0)
                                                }

        for k, v in self.intersection_coordinators.items():
            self.intersection_coordinators[k] = carla.Location(v[0], v[1], v[2])
        # for i, intersection in enumerate(self.intersection_coordinators):
        #     self.intersection_coordinators[i] = carla.Location(intersection[0], intersection[1], intersection[2])

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def save_ego_data(self, path):
        with open(os.path.join(path, 'ego_data.json'), 'w') as f:
            json.dump(self.ego_data, f, indent=4)
        self.ego_data = {}

    def record_speed_control_transform(self, world, frame):
        v = world.player.get_velocity()
        c = world.player.get_control()
        t = world.player.get_transform()
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

    def tick(self, world, clock, camera, frame, display, root_path):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        vehicles = world.world.get_actors().filter('vehicle.*')
        peds = world.world.get_actors().filter('walker.pedestrian.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
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
        self._info_text += [
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        moving = False
        acc = world.player.get_acceleration().length()

        if acc > 1.0 and c.throttle > 0:
            moving = True
            # print(acc)

        if len(vehicles) > 2:

            num_actors = 0
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            peds = [(distance(x.get_location()), x) for x in peds if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 31.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
                num_actors += 1

            for d, ped in sorted(peds, key=lambda peds: peds[0]):
                if d > 35.0:
                    break
                # vehicle_type = get_actor_display_name(ped, truncate=22)
                self._info_text.append('% 4dm pedestrian' % (d))
                num_actors += 1

            d = self.d_2_intersection
            in_intersection = False
            topo_id = ''
            for k, location in self.intersection_coordinators.items():
                if world.player.get_location().distance(location) < d:
                    d = world.player.get_location().distance(location)
                    in_intersection = True
                    topo_id = k
                    # print(topo_id)
            self._info_text.append('intersection in % 4dm' % (d))


           

            

            leave = False
            if d > self.d_last:
                leave = True
                # print('leaving')
            # if num_actors > 2 and num_actors < 7 and not self.recording and in_intersection and not leave:
            if num_actors > 2 and not self.recording and in_intersection and not leave and moving:
                # print(d)
            # if num_actors > 3 and not self.recording:
                self.recording = True
                self.recording_time = time.time()

                stored_path = os.path.join(root_path, topo_id)
                if not os.path.exists(stored_path) :
                    os.makedirs(stored_path)
                stored_path = os.path.join(stored_path, 'variant_scenario')
                if not os.path.exists(stored_path) :
                    os.makedirs(stored_path)

                folders = [f for f in os.listdir(stored_path) if os.path.isdir(os.path.join(stored_path, f))]

                if len(folders) == 0:
                    v_id = str(1)
                else:
                    for i in range(len(folders)):
                        folders[i] = int(folders[i])
                    v_id = str(int(max(folders))+1)
                stored_path = os.path.join(stored_path, v_id)
                if not os.path.exists(stored_path) :
                    os.makedirs(stored_path)
                print(stored_path)
                self.v_id = v_id
                self.stored_path = stored_path


                # town/topo_id/variant/ins_id

                self.out = cv2.VideoWriter(stored_path+"/" + str(v_id) +".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20,  (1536, 512) )
                camera.toggle_recording(stored_path)
                print(stored_path+"/" + str(self.v_id) +".mp4")
            elif self.recording and time.time() - self.recording_time > 10.0:
                camera.toggle_recording(self.stored_path)
                self.save_ego_data(self.stored_path)
                self.recording = False
                self.out.release()
                world.next_weather()
                self.stored_path = root_path

            self.d_last = d

        if self.recording:
            self.record_speed_control_transform(world, frame)
            self.out.write(display)
            self._info_text += ['recording vedio:']

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
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


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
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
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
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
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
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction, save_mode=True):
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
        # self.flow = []
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

        # self.flow_bp = bp_library.find('sensor.camera.optical_flow')
        # self.flow_bp.set_attribute('image_size_x', str(1236))
        # self.flow_bp.set_attribute('image_size_y', str(512))
        # self.flow_bp.set_attribute('fov', str(120.0))
        self.front_cam_bp.set_attribute('lens_circle_multiplier', '0.0')
        self.front_cam_bp.set_attribute('lens_circle_falloff', '0.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_intensity', '3.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_offset', '500')
        # self.flow_bp.set_attribute('focal_distance', str(500))
        # if self.flow_bp.has_attribute('gamma'):
        #     self.flow_bp.set_attribute('gamma', str(gamma_correction))
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
                # self.sensor_flow = self._parent.get_world().spawn_actor(
                #     self.flow_bp,
                #     self._camera_transforms[0][0],
                #     attach_to=self._parent,
                #     attachment_type=self._camera_transforms[0][1])
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
                # self.sensor_flow.listen(lambda image: CameraManager._parse_image(weak_self, image, 'flow'))
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
            # t_flow = Process(target=self.save_img, args=(self.flow, 8, path, 'flow'))
            t_lidar = Process(target=self.save_img, args=(self.lidar, 6, path, 'lidar'))
            start_time = time.time()

            # t_lbc_img.start()
            t_front.start()
            t_lbc_ins.start()
            t_ins_front.start()
            t_depth_front.start()
            # t_flow.start()
            t_lidar.start()
            
            # t_lbc_img.join()
            t_front.join()
            t_lbc_ins.join()
            t_ins_front.join()
            t_depth_front.join()
            # t_flow.join()
            t_lidar.join()
            
            # self.lbc_img = []
            self.lbc_ins = []
            self.front_img = []
            self.front_ins = []
            self.front_depth = []
            # self.flow = []
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
            # elif view == 'flow':
            #     self.flow.append(image)
            elif view == 'seg_front':
                self.front_seg.append(image)
            elif view == 'ins_front':
                self.front_ins.append(image)

            elif view == 'depth_front':
                self.front_depth.append(image)
            
def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()


        stored_path = os.path.join('data_collection', 'ap_'+ args.town)
        if not os.path.exists(stored_path) :
            os.makedirs(stored_path)


        hud = HUD(args.width, args.height, args.distance, args.town, stored_path)
        world = World(client.load_world(args.town), hud, args)

        lights = []
        light_actors = world.world.get_actors().filter('traffic.traffic_light*')
        for l in light_actors:
            lights.append(l)
        light_dict, light_transform_dict = read_traffic_lights(lights)


        settings = world.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True  # Enables synchronous mode
        world.world.apply_settings(settings)

        controller = KeyboardControl(world, args.autopilot)
        vehicles_list, all_actors, all_id = spawn_actor_nearby(world.world, client, 300, 0.4, 120)

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(40)
            frame = world.world.tick()

            ref_light = get_next_traffic_light(
                world.player, world.world, light_transform_dict)
            annotate = annotate_trafficlight_in_group(
                ref_light, lights, world.world)
            if annotate:
                now = datetime.datetime.now()
                n_min = now.minute
                r = n_min%3
                set_light_state(light_dict, annotate, r)
            if controller.parse_events(client, world, clock):
                return

            pygame.image.save(display, os.path.join(stored_path, "screenshot.jpeg"))
            image = cv2.imread(os.path.join(stored_path, "screenshot.jpeg"))
            

            world.tick(clock, frame, image, stored_path)
            world.render(display)
            #print("location: ",world.player.get_location())
            pygame.display.flip()

    finally:
        world.world.settings.synchronous_mode = False

        print('destroying vehicles')
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        # for i in range(0, len(all_id), 2):
        #     all_actors[i].stop()

        # print('destroying walkers')
        # client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)


        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()



        pygame.quit()


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
        default='model3',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--distance',
        default=25.0,
        type=float,
        help='distance to intersection for toggling camera)')
    argparser.add_argument(
        '--town',
        default='Town05',
        type=str,
        help='map)')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

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
