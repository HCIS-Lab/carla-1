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

try:
    #
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
       sys.version_info.major,
       sys.version_info.minor,
       'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    #
    sys.path.append('../carla/agents/navigation')
    sys.path.append('../carla/agents')
    sys.path.append('../carla/')
    sys.path.append('../../HDMaps')

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
import xml.etree.ElementTree as ET
import carla_vehicle_annotator as cva
import matplotlib.pyplot as plt


from random_actors import spawn_actor_nearby

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


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, client_bp, hud, args):
        self.world = carla_world
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True # Enables synchronous mode
        self.world.apply_settings(settings)
        self.actor_role_name = args.rolename
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

        self.speed_list = []
        self.control_list = []
        self.transform_list = []

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(
            self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
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
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.camera_manager.background = True

        
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
        self.speed_list.append([frame, v.x, v.y, v.z])
        self.control_list.append([frame, c.throttle, c.steer, c.brake, 
                                c.hand_brake, c.manual_gear_shift, c.gear])
        self.transform_list.append([t.location.x, t.location.y, t.location.z, 
                                    t.rotation.pitch, t.rotation.yaw, t.rotation.roll])

    def save_speed_control_transform(self, path):
        speed = np.asarray(self.speed_list)
        control = np.asarray(self.control_list)
        transform = np.asarray(self.transform_list)

        np.save('%s/speed' % (path), speed)
        np.save('%s/control' % (path), control)
        np.save('%s/transform' % (path), transform)
        self.speed_list = []
        self.control_list = []
        self.transform_list = []

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
            self.camera_manager.sensor_top,
            # self.camera_manager.sensor_front,
            # self.camera_manager.sensor_left,
            # self.camera_manager.sensor_right,
            # self.camera_manager.sensor_back,
            # self.camera_manager.sensor_back_left,
            # self.camera_manager.sensor_back_right,
            # self.camera_manager.sensor_lidar,
            # self.camera_manager.sensor_dvs,
            # self.camera_manager.sensor_flow,
            # self.camera_manager.seg_top,
            # self.camera_manager.seg_front,
            # self.camera_manager.seg_back,
            # self.camera_manager.seg_right,
            # self.camera_manager.seg_left,
            # self.camera_manager.seg_back_right,
            # self.camera_manager.seg_back_left,
            # self.camera_manager.depth_front,
            # self.camera_manager.depth_right,
            # self.camera_manager.depth_left,
            # self.camera_manager.depth_back,
            # self.camera_manager.depth_back_right,
            # self.camera_manager.depth_back_left,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        self.camera_manager.sensor_front = None
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
                    scenario_name=None
                    world.camera_manager.recording= not  world.camera_manager.recording
                    # world.lidar_sensor.recording= not  world.lidar_sensor.recording
                    # if not  world.lidar_sensor.recording:
                    if not world.camera_manager.recording:
                        scenario_name=input("scenario id: ")
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
            #world.player.apply_control(self._control)
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
    def __init__(self, width, height):
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

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        # if len(self.history) > 4000:
        #     self.history.pop(0)
        self.collision = True

    def save_history(self, path):
        if self.collision:
            for i, collision in enumerate(self.history):
                self.history[i] = list(self.history[i])
            history = np.asarray(self.history)
            if len(history) != 0:
                np.save('%s/collision_history' % (path), history)

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
    def __init__(self, parent_actor):
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
        self.recording = False
        self.imu_save = []
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
            self.imu_save.append([sensor_data.frame, 
                                self.accelerometer[0], self.accelerometer[1], self.accelerometer[2],
                                self.gyroscope[0], self.gyroscope[1], self.gyroscope[2], 
                                self.compass])

    def toggle_recording_IMU(self, path):
        self.recording = not self.recording
        if not self.recording:
            t_top = threading.Thread(target = self.save_IMU, args=(self.imu_save, path))
            t_top.start()
            self.imu_save = []

    def save_IMU(self, save_list, path):
        np_imu = np.asarray(save_list)
        np.save('%s/imu' % (path), np_imu)
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
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor_top = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False

        self.top_img = []
        self.front_img = []
        self.left_img = []
        self.right_img = []
        self.back_img = []
        self.back_left_img = []
        self.back_right_img = []

        self.lidar = []
        self.flow = []
        self.dvs = []

        # self.top_iseg = []
        self.top_seg = []
        self.front_seg = []
        self.left_seg = []
        self.right_seg = []
        self.back_seg = []
        self.back_left_seg = []
        self.back_right_seg = []

        self.front_depth = []
        self.left_depth = []
        self.right_depth = []
        self.back_depth = []
        self.back_left_depth = []
        self.back_right_depth = []

        self.bbox = []
        self.img_dict = {}
        self.snap_dict = {}

        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                # front view
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                # front-left view
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z), carla.Rotation(yaw=-55)), Attachment.Rigid),
                # front-right view
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z), carla.Rotation(yaw=55)), Attachment.Rigid),
                # back view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z), carla.Rotation(yaw=180)), Attachment.Rigid),
               # back-left view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z), carla.Rotation(yaw=235)), Attachment.Rigid),
                # back-right view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z), carla.Rotation(yaw=-235)), Attachment.Rigid),
                # top view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y, z=23*bound_z), carla.Rotation(pitch=18.0)), Attachment.SpringArm)
                ]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.optical_flow', None, 'Optical Flow', {}],
            ['sensor.other.lane_invasion', None, 'Lane lane_invasion',{}],
            ['sensor.camera.instance_segmentation', cc.CityScapesPalette, 'Camera Instance Segmentation (CityScapes Palette)', {}],

            ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
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
            if self.sensor_top is not None:
                self.sensor_top.destroy()
                self.surface = None

                # rgb sensor
            self.sensor_top = self._parent.get_world().spawn_actor(
                self.sensors[0][-1],
                self._camera_transforms[6][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[6][1])
            # self.sensor_front = self._parent.get_world().spawn_actor(
            #     self.sensors[0][-1],
            #     self._camera_transforms[0][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[0][1])
            # self.sensor_left = self._parent.get_world().spawn_actor(
            #     self.sensors[0][-1],
            #     self._camera_transforms[1][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[1][1])
            # self.sensor_right = self._parent.get_world().spawn_actor(
            #     self.sensors[0][-1],
            #     self._camera_transforms[2][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[2][1])
            # self.sensor_back = self._parent.get_world().spawn_actor(
            #     self.sensors[0][-1],
            #     self._camera_transforms[3][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[3][1])
            # self.sensor_back_left = self._parent.get_world().spawn_actor(
            #     self.sensors[0][-1],
            #     self._camera_transforms[4][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[4][1])
            # self.sensor_back_right = self._parent.get_world().spawn_actor(
            #     self.sensors[0][-1],
            #     self._camera_transforms[5][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[5][1])

            # # lidar sensor
            # self.sensor_lidar = self._parent.get_world().spawn_actor(
            #     self.sensors[6][-1],
            #     self._camera_transforms[0][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[0][1])

            # self.sensor_dvs = self._parent.get_world().spawn_actor(
            #     self.sensors[7][-1],
            #     self._camera_transforms[0][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[0][1])

            # # optical flow
            # self.sensor_flow = self._parent.get_world().spawn_actor(
            #     self.sensors[8][-1],
            #     self._camera_transforms[0][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[0][1])

            # # segmentation sensor
            # # self.iseg_top = self._parent.get_world().spawn_actor(
            # #     self.sensors[-1][-1],
            # #     self._camera_transforms[6][0],
            # #     attach_to=self._parent,
            # #     attachment_type=self._camera_transforms[6][1])

            # self.seg_top = self._parent.get_world().spawn_actor(
            #     self.sensors[5][-1],
            #     self._camera_transforms[6][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[6][1])
            # self.seg_front = self._parent.get_world().spawn_actor(
            #     self.sensors[5][-1],
            #     self._camera_transforms[0][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[0][1])
            # self.seg_left = self._parent.get_world().spawn_actor(
            #     self.sensors[5][-1],
            #     self._camera_transforms[1][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[1][1])
            # self.seg_right = self._parent.get_world().spawn_actor(
            #     self.sensors[5][-1],
            #     self._camera_transforms[2][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[2][1])
            # self.seg_back = self._parent.get_world().spawn_actor(
            #     self.sensors[5][-1],
            #     self._camera_transforms[3][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[3][1])
            # self.seg_back_left = self._parent.get_world().spawn_actor(
            #     self.sensors[5][-1],
            #     self._camera_transforms[4][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[4][1])
            # self.seg_back_right = self._parent.get_world().spawn_actor(
            #     self.sensors[5][-1],
            #     self._camera_transforms[5][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[5][1])

            # # depth estimation sensor
            # self.depth_front = self._parent.get_world().spawn_actor(
            #     self.sensors[2][-1],
            #     self._camera_transforms[0][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[0][1])
            # self.depth_left = self._parent.get_world().spawn_actor(
            #     self.sensors[2][-1],
            #     self._camera_transforms[1][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[1][1])
            # self.depth_right = self._parent.get_world().spawn_actor(
            #     self.sensors[2][-1],
            #     self._camera_transforms[2][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[2][1])
            # self.depth_back = self._parent.get_world().spawn_actor(
            #     self.sensors[2][-1],
            #     self._camera_transforms[3][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[3][1])
            # self.depth_back_left = self._parent.get_world().spawn_actor(
            #     self.sensors[2][-1],
            #     self._camera_transforms[4][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[4][1])
            # self.depth_back_right = self._parent.get_world().spawn_actor(
            #     self.sensors[2][-1],
            #     self._camera_transforms[5][0],
            #     attach_to=self._parent,
            #     attachment_type=self._camera_transforms[5][1])

            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor_top.listen(lambda image: CameraManager._parse_image(weak_self, image, 'top'))
            # self.sensor_front.listen(lambda image: CameraManager._parse_image(weak_self, image, 'front'))
            # self.sensor_right.listen(lambda image: CameraManager._parse_image(weak_self, image, 'right'))
            # self.sensor_left.listen(lambda image: CameraManager._parse_image(weak_self, image, 'left'))
            # self.sensor_back.listen(lambda image: CameraManager._parse_image(weak_self, image, 'back'))
            # self.sensor_back_right.listen(lambda image: CameraManager._parse_image(weak_self, image, 'back_right'))
            # self.sensor_back_left.listen(lambda image: CameraManager._parse_image(weak_self, image, 'back_left'))

            # self.sensor_lidar.listen(lambda image: CameraManager._parse_image(weak_self, image, 'lidar'))
            # self.sensor_dvs.listen(lambda image: CameraManager._parse_image(weak_self, image, 'dvs'))
            # self.sensor_flow.listen(lambda image: CameraManager._parse_image(weak_self, image, 'flow'))
            
            # # self.iseg_top.listen(lambda image: CameraManager._parse_image(weak_self, image, 'iseg_top'))
            # self.seg_top.listen(lambda image: CameraManager._parse_image(weak_self, image, 'seg_top'))
            # self.seg_front.listen(lambda image: CameraManager._parse_image(weak_self, image, 'seg_front'))
            # self.seg_right.listen(lambda image: CameraManager._parse_image(weak_self, image, 'seg_right'))
            # self.seg_left.listen(lambda image: CameraManager._parse_image(weak_self, image, 'seg_left'))
            # self.seg_back.listen(lambda image: CameraManager._parse_image(weak_self, image, 'seg_back'))
            # self.seg_back_right.listen(lambda image: CameraManager._parse_image(weak_self, image, 'seg_back_right'))
            # self.seg_back_left.listen(lambda image: CameraManager._parse_image(weak_self, image, 'seg_back_left'))

            # self.depth_front.listen(lambda image: CameraManager._parse_image(weak_self, image, 'depth_front'))
            # self.depth_right.listen(lambda image: CameraManager._parse_image(weak_self, image, 'depth_right'))
            # self.depth_left.listen(lambda image: CameraManager._parse_image(weak_self, image, 'depth_left'))
            # self.depth_back.listen(lambda image: CameraManager._parse_image(weak_self, image, 'depth_back'))
            # self.depth_back_right.listen(lambda image: CameraManager._parse_image(weak_self, image, 'depth_back_right'))
            # self.depth_back_left.listen(lambda image: CameraManager._parse_image(weak_self, image, 'depth_back_left'))

        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self, path):
        self.recording = not self.recording
        if not self.recording:
            t_top = threading.Thread(target = self.save_img,args=(self.top_img, 0, path, 'top'))
            t_front = threading.Thread(target = self.save_img,args=(self.front_img, 0, path, 'front'))
            t_right = threading.Thread(target = self.save_img,args=(self.right_img, 0, path, 'right'))
            t_left = threading.Thread(target = self.save_img,args=(self.left_img, 0, path, 'left'))
            t_back = threading.Thread(target = self.save_img,args=(self.back_img, 0, path, 'back'))
            t_back_right = threading.Thread(target = self.save_img,args=(self.back_right_img, 0, path, 'back_right'))
            t_back_left = threading.Thread(target = self.save_img,args=(self.back_left_img, 0, path, 'back_left'))

            t_lidar = threading.Thread(target = self.save_img,args=(self.lidar, 6, path, 'lidar'))
            t_dvs = threading.Thread(target = self.save_img,args=(self.dvs, 7, path, 'dvs'))
            t_flow = threading.Thread(target = self.save_img,args=(self.flow, 8, path, 'flow'))

            # t_iseg_top = threading.Thread(target = self.save_img, args=(self.top_seg, 10, path, 'iseg_top'))
            t_seg_top = threading.Thread(target = self.save_img, args=(self.top_seg, 5, path, 'seg_top'))
            t_seg_front = threading.Thread(target = self.save_img, args=(self.front_seg, 5, path, 'seg_front'))
            t_seg_right = threading.Thread(target = self.save_img, args=(self.right_seg, 5, path, 'seg_right'))
            t_seg_left = threading.Thread(target = self.save_img, args=(self.left_seg, 5, path, 'seg_left'))
            t_seg_back = threading.Thread(target = self.save_img, args=(self.back_seg, 5, path, 'seg_back'))
            t_seg_back_right = threading.Thread(target = self.save_img, args=(self.back_right_seg, 5, path, 'seg_back_right'))
            t_seg_back_left = threading.Thread(target = self.save_img, args=(self.back_left_seg, 5, path, 'seg_back_left'))

            t_depth_front = threading.Thread(target = self.save_img, args=(self.front_depth, 1, path, 'depth_front'))
            t_depth_right = threading.Thread(target = self.save_img, args=(self.right_depth, 1, path, 'depth_right'))
            t_depth_left = threading.Thread(target = self.save_img, args=(self.left_depth, 1, path, 'depth_left'))
            t_depth_back = threading.Thread(target = self.save_img, args=(self.back_depth, 1, path, 'depth_back'))
            t_depth_back_right = threading.Thread(target = self.save_img, args=(self.back_right_depth, 1, path, 'depth_back_right'))
            t_depth_back_left = threading.Thread(target = self.save_img, args=(self.back_left_depth, 1, path, 'depth_back_left'))

#             t_bbox = threading.Thread(target = self.save_bbox, args=(self.bbox, path))

            t_top.start()
            t_front.start()
            t_left.start()
            t_right.start()
            t_back.start()
            t_back_left.start()
            t_back_right.start()

            t_lidar.start()
            t_dvs.start()
            t_flow.start()

            # t_iseg_top.start()
            t_seg_top.start()
            t_seg_front.start()
            t_seg_right.start()
            t_seg_left.start()
            t_seg_back.start()
            t_seg_back_right.start()
            t_seg_back_left.start()

            t_depth_front.start()
            t_depth_right.start()
            t_depth_left.start()
            t_depth_back.start()
            t_depth_back_right.start()
            t_depth_back_left.start()

#             t_bbox.start()

            self.top_img = []
            self.front_img = []
            self.right_img = []
            self.left_img = []
            self.back_img = []
            self.back_right_img = []
            self.back_left_img = []

            self.lidar = []
            self.dvs = []
            self.flow = []

            # self.top_iseg = []
            self.top_seg = []
            self.front_seg = []
            self.right_seg = []
            self.left_seg = []
            self.back_seg = []
            self.back_right_seg = []
            self.back_left_seg = []

            self.front_depth = []
            self.right_depth = []
            self.left_depth = []
            self.back_depth = []
            self.back_right_depth = []
            self.back_left_depth = []

            t_depth_front.join()
            self.save_bbox([self.bbox,self.img_dict,self.snap_dict],path)

            self.bbox = []
            self.img_dict = {}
            self.snap_dict = {}

        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def save_img(self, img_list, sensor, path, view='top'):
        for img in img_list:
            if img.frame%1 == 0:
                if 'seg' in view:
                    img.save_to_disk('%s/%s/%s/%08d' % (path, self.sensors[sensor][2], view, img.frame), cc.CityScapesPalette)
                elif 'depth' in view:
                    img.save_to_disk('%s/%s/%s/%08d' % (path, self.sensors[sensor][2], view, img.frame), cc.LogarithmicDepth)
                elif 'dvs' in view:
                    dvs_events = np.frombuffer(img.raw_data, dtype=np.dtype([
                        ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
                    dvs_img = np.zeros((img.height, img.width, 3), dtype=np.uint8)
                    # Blue is positive, red is negative
                    dvs_img[dvs_events[:]['y'], dvs_events[:]
                    ['x'], dvs_events[:]['pol'] * 2] = 255
                    # img = img.to_image()
                    stored_path = os.path.join(path, self.sensors[sensor][2], view)
                    if not os.path.exists(stored_path):
                        os.makedirs(stored_path)
                    np.save('%s/%08d' % (stored_path, img.frame), dvs_img)
                elif 'flow' in view:
                    frame = img.frame
                    img = img.get_color_coded_flow()
                    array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (img.height, img.width, 4))
                    array = array[:, :, :3]
                    array = array[:, :, ::-1]
                    stored_path = os.path.join(path, self.sensors[sensor][2], view)
                    if not os.path.exists(stored_path):
                        os.makedirs(stored_path)
                    np.save('%s/%08d' % (stored_path, frame), array)
                else:
                    img.save_to_disk('%s/%s/%s/%08d' % (path, self.sensors[sensor][2], view, img.frame))
        print("%s %s save finished." % (self.sensors[sensor][2], view))

    def save_bbox(self,input_list,path):
        VIEW_WIDTH = int(self.sensor_front.attributes['image_size_x'])
        VIEW_HEIGHT = int(self.sensor_front.attributes['image_size_y'])
        VIEW_FOV = int(float(self.sensor_front.attributes['fov']))
        bbox,img_dict,snap_dict = input_list
        for index,item in enumerate(bbox):
            depth_img = item
            try:
                vehicles,cam = snap_dict[depth_img.frame]
            except:
                continue
            rgb_img = img_dict[depth_img.frame]
            depth_meter = cva.extract_depth(depth_img)
            filtered, removed =  cva.auto_annotate(vehicles, cam, depth_meter,VIEW_WIDTH,VIEW_HEIGHT,VIEW_FOV)
            cva.save_output(rgb_img, filtered['bbox'], path,filtered['vehicles'], removed['bbox'], removed['vehicles'], save_patched=True, out_format='json')
    

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

        if self.recording and image.frame%1 == 0:
            if view == 'top':
                self.top_img.append(image)
            elif view == 'front':
                self.front_img.append(image)
                self.img_dict[image.frame]=image
            elif view == 'left':
                self.left_img.append(image)
            elif view == 'right':
                self.right_img.append(image)
            elif view == 'back':
                self.back_img.append(image)
            elif view == 'back_left':
                self.back_left_img.append(image)
            elif view == 'back_right':
                self.back_right_img.append(image)

            elif view == 'lidar':
                self.lidar.append(image)
            elif view == 'dvs':
                self.dvs.append(image)
            elif view == 'flow':
                self.flow.append(image)

            # elif view == 'iseg_top':
            #     self.top_iseg.append(image)

            elif view == 'seg_top':
                self.top_seg.append(image)
            elif view == 'seg_front':
                self.front_seg.append(image)
            elif view == 'seg_right':
                self.right_seg.append(image)
            elif view == 'seg_left':
                self.left_seg.append(image)
            elif view == 'seg_back':
                self.back_seg.append(image)
            elif view == 'seg_back_right':
                self.back_right_seg.append(image)
            elif view == 'seg_back_left':
                self.back_left_seg.append(image)

            elif view == 'depth_front':
                self.front_depth.append(image)
                if self.sensor_front is not None:#self.img_dict[image.frame] is not None and self.sensor_front is not None:
                    world = self._parent.get_world()
                    snapshot = world.get_snapshot()
                    actors = world.get_actors()
                    try:
                        sensor_transform = snapshot.find(self.sensor_front.id).get_transform()
                        vehicles = cva.snap_processing(actors.filter('vehicle.*'), snapshot)
                        vehicles+= cva.snap_processing(actors.filter('walker.*'), snapshot)
                        self.snap_dict[snapshot.frame] = [vehicles,sensor_transform]
                        self.bbox.append(image)
                    except:
                        print("Initial frame.")
            elif view == 'depth_right':
                self.right_depth.append(image)
            elif view == 'depth_left':
                self.left_depth.append(image)
            elif view == 'depth_back':
                self.back_depth.append(image)
            elif view == 'depth_back_right':
                self.back_right_depth.append(image)
            elif view == 'depth_back_left':
                self.back_left_depth.append(image)   


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


def get_transform(np_transform):
    transform = carla.Transform(Location(np_transform[0], np_transform[1], np_transform[2]),
                                Rotation(np_transform[3], np_transform[4], np_transform[5]))
    return transform


def read_control(path='control.npy'):
    """ param:

    """
    control = np.load(path)
    control_list = []
    init_transform = control[0]
    init_transform = carla.Transform(Location(x=control[0][0], y=control[0][1], z=control[0][2]+1),
                                     Rotation(pitch=control[0][3], yaw=control[0][4], roll=control[0][5]))
    for i in range(1, len(control)):
        control_list.append(carla.VehicleControl(float(control[i][0]), float(control[i][1]), float(control[i][2]), bool(control[i][3]),
                                                 bool(control[i][4]), bool(control[i][5]), int(control[i][6])))

    return init_transform, control_list


def read_transform(path='control.npy'):
    """ param:

    """
    transform_npy = np.load(path)
    transform_list = []
    for i, transform in enumerate(transform_npy):
        if i == 0:
            transform_list.append(carla.Transform(Location(x=transform[0], y=transform[1], z=transform[2]+1),
                                                  Rotation(pitch=transform[3], yaw=transform[4], roll=transform[5])))
        else:
            transform_list.append(carla.Transform(Location(x=transform[0], y=transform[1], z=transform[2]),
                                                  Rotation(pitch=transform[3], yaw=transform[4], roll=transform[5])))

    return transform_list


def read_velocity(path='velocity.npy'):
    velocity_npy = np.load(path)
    velocity_list = []
    for velocity in velocity_npy:
        v = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2)**0.5
        velocity_list.append(v)
        # velocity_list.append(carla.Vector3D(x=velocity[0], y=velocity[1], z=velocity[2]))

    return velocity_list

def read_traffic_lights(path, lights):
    
    path = os.path.join(path, 'traffic_light')
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    light_dict = dict()
    for i, f in enumerate(files):
        l_id = int(f.split('.npy')[0])
        light_state = np.load(os.path.join(path, f), allow_pickle=True)
        new_light_state = []
        for iter, state in enumerate(light_state):
            if iter == 0:
                l_loc = carla.Location(float(state[0]), float(state[1]), float(state[2]))
            if state[0] == 'Red':
                new_light_state.append(carla.TrafficLightState.Red)
            elif state[0] == 'Yellow':
                new_light_state.append(carla.TrafficLightState.Yellow)
            elif state[0] == 'Green':
                new_light_state.append(carla.TrafficLightState.Green)
            elif state[0] == 'Off':
                new_light_state.append(carla.TrafficLightState.Off)
            else:
                new_light_state.append(carla.TrafficLightState.Unknown)
        # if i == 0:
        #     min_len = light.shape[0]-1
        # elif light.shape[0]-1 < min_len:
        #     min_len = light.shape[0] -1
        # light_id  = int(f.split('.')[0])
        # l_loc = carla.Location(light[0][0], light[0][1], light[0][2])
        min_d = 500.0
        for new_l in lights:
            if new_l.get_location().distance(l_loc) < min_d:
                min_d = new_l.get_location().distance(l_loc)
                new_id = new_l.id
        light_dict[new_id] = new_light_state
        # light_dict[l_id] = light_state
    # print(len(files))
    # print(len(lights))
    # print(len(light_dict))

    return light_dict
    # return light_dict, min_len

def set_light_state(lights, light_dict, index):
    for l in lights:
        if l.id in light_dict:
            index = index if len(light_dict[l.id]) > index else -1
            state = light_dict[l.id][index]
            l.set_state(state)

def control_with_trasform_controller(controller, transform):
    control_signal = controller.run_step(10, transform)
    return control_signal

def auto_spawn_object(world,second):
    this_map=world.world.get_map()
    new_obj=None
    try:
        bp_list=world.world.get_blueprint_library().filter('static')
        while True:
            time.sleep(second)
            if new_obj is not None:
                new_obj.destroy()
                new_obj=None
            if world.player.is_at_traffic_light():
                continue
            waypoint = this_map.get_waypoint(world.player.get_location(),lane_type=carla.LaneType.Shoulder)
            if waypoint is None:
                continue
            waypoint_list=waypoint.next(15)
            if waypoint_list:
                waypoint = waypoint_list[0]

            obj_bp=random.choice(bp_list)
            new_obj=world.world.try_spawn_actor(obj_bp, waypoint.transform)#carla.Transform(new_obj_location, vehicle_rotation))
            if new_obj!=None:
                print("Spawn object.")
    finally:
        if new_obj is not None:
            new_obj.destroy()

def collect_trajectory(get_world, agent, scenario_id, period_end):
    filepath = "data_collection/"
    filepath = filepath + str(scenario_id) + '/' + str(scenario_id) + '.csv'
    is_exist = os.path.isfile(filepath)
    f = open(filepath, 'a+')
    w = csv.writer(f)

    actors = get_world.world.get_actors()
    town_map = get_world.world.get_map()
    
    if not is_exist:
        w.writerow(['TIMESTAMP', 'TRACK_ID', 'OBJECT_TYPE', 'X', 'Y', 'CITY_NAME'])
    agent_id = agent.id

    period_start = 0
    
    time_start = time.time()
    try:
        while True:
            time_end = time.time()
            if period_start < period_end:
                if (time_end - time_start) > 0.1:
                    period_start += 0.1
                    #print(period_start, period_end)
                    time_start = time.time()
                    for actor in actors:
                        if agent_id == actor.id:
                            agent = actor
                        if agent.get_location().x == 0 and agent.get_location().y == 0:
                            return true
                        if actor.type_id[0:7] == 'vehicle' or actor.type_id[0:6] == 'walker':
                            x = actor.get_location().x 
                            y = actor.get_location().y
                            id = actor.id
                            if x == agent.get_location().x and y == agent.get_location().y:
                                w.writerow([time_start-10**9, id, 'AGENT', str(x), str(y), town_map.name])
                            else:
                                if ((x - agent.get_location().x)**2 + (y - agent.get_location().y)**2) < 75**2:
                                    if actor.type_id[0:7] == 'vehicle':
                                        w.writerow([time_start-10**9, id, 'vehicle', str(x), str(y), town_map.name])
                                    elif actor.type_id[0:6] == 'walker':
                                        w.writerow([time_start-10**9, id, 'walker', str(x), str(y), town_map.name])
            else:
                return False
    except:
        print("trajectory_collection finished")

def collect_topology(get_world, agent, scenario_id, t):
    filepath = "data_collection/"
    town_map = get_world.world.get_map()
    if not os.path.exists('data_collection/%s/topology/'% (scenario_id)):
        os.mkdir('data_collection/%s/topology/'% (scenario_id))
    with open(filepath + scenario_id + '/scenario_description.json') as f:
        data = json.load(f)
    time_start = time.time()
    try:
        while True:
            time_end = time.time()
            if (time_end - time_start) > t:         # may need change 
                waypoint = town_map.get_waypoint(agent.get_location())
                waypoint_list = town_map.generate_waypoints(2.0)
                nearby_waypoint = []
                roads = []
                all = []
                for wp in waypoint_list:
                    dist_x = int(wp.transform.location.x) - int(waypoint.transform.location.x)
                    dist_y = int(wp.transform.location.y) - int(waypoint.transform.location.y)
                    if abs(dist_x) <= 37.5 and abs(dist_y) <= 37.5:
                        nearby_waypoint.append(wp)
                        roads.append((wp.road_id, wp.lane_id))
                for wp in nearby_waypoint:
                    for id in roads:
                        if wp.road_id == id[0] and wp.lane_id == id[1]:
                            all.append(((wp.road_id, wp.lane_id), wp))
                            break
                all = sorted(all, key = lambda s: s[0][1])
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
                    halluc_lane_1, halluc_lane_2 = np.empty((0, 3*2)), np.empty((0, 3*2))
                    is_traffic_control = False
                    is_junction = False
                    for k in range(len(all)-1):
                        if (i, j) == all[k][0] and (i, j) == all[k+1][0]:
                            if all[k][1].get_landmarks(50, False):      # may change & need traffic light
                                is_traffic_control = True
                            if all[k][1].is_junction:                   
                                is_junction = True
                            # -= norm center
                            before = [all[k][1].transform.location.x, all[k][1].transform.location.y]
                            after = [all[k+1][1].transform.location.x, all[k+1][1].transform.location.y]
                            distance = []
                            for t in range(len(before)):
                                distance.append(after[t] - before[t])
                            np_distance = np.array(distance)
                            norm = np.linalg.norm(np_distance)
                            e1, e2 = rotate_quat @ np_distance / norm, rotate_quat.T @ np_distance / norm
                            lane_1 = np.hstack((before + e1 * all[k][1].lane_width/2, all[k][1].transform.location.z, after + e1 * all[k][1].lane_width/2, all[k+1][1].transform.location.z))
                            lane_2 = np.hstack((before + e2 * all[k][1].lane_width/2, all[k][1].transform.location.z, after + e2 * all[k][1].lane_width/2, all[k+1][1].transform.location.z))
                            halluc_lane_1 = np.vstack((halluc_lane_1, lane_1))
                            halluc_lane_2 = np.vstack((halluc_lane_2, lane_2))
                    if data['traffic_light']:
                        is_junction = True
                    lane_feature_ls.append([halluc_lane_1, halluc_lane_2, is_traffic_control, is_junction, (i, j)])           
                np.save('data_collection/' + str(scenario_id) + '/topology/' + str(scenario_id), np.array(lane_feature_ls))

                # Other objects can also be plot in the graph
                #with open(filepath + str(scenario_id) + '/' + str(scenario_id) + '.csv', newline='') as csvfile:
                #    rows = csv.DictReader(csvfile)

                for features in lane_feature_ls:
                    xs, ys = np.vstack((features[0][:, :2], features[0][-1, 3:5]))[:, 0], np.vstack((features[0][:, :2], features[0][-1, 3:5]))[:, 1]
                    plt.plot(xs, ys, '--', color='red')
                    x_s, y_s = np.vstack((features[1][:, :2], features[1][-1, 3:5]))[:, 0], np.vstack((features[1][:, :2], features[1][-1, 3:5]))[:, 1]
                    plt.plot(x_s, y_s, '--', color='blue')
                plt.savefig('data_collection/' + str(scenario_id) + '/topology/topology.png')
                break
        return False 
    except:
        print("topology_collection finished")

def set_bp(blueprint, actor_id):
    blueprint = random.choice(blueprint)
    blueprint.set_attribute('role_name', actor_id)
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

def save_description(world, args, stored_path, weather):
    vehicles = world.world.get_actors().filter('vehicle.*')
    peds = world.world.get_actors().filter('walker.*')
    d = dict()
    d['num_actor'] = len(vehicles) + len(peds)
    d['num_vehicle'] = len(vehicles)
    d['weather'] = str(weather)
    d['noise_trajectory'] = args.noise_trajectory
    d['random_objects'] = args.random_objects
    d['random_actors'] = args.random_actors
    d['simulation_time'] = int(world.hud.simulation_time)

    with open('%s/dynamic_description.json' % (stored_path), 'w') as f:
        json.dump(d, f)

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    path = 'data_collection/'+str(args.scenario_id)

    filter_dict = {}
    try:
        for root, _, files in os.walk(path + '/filter/'):
            for name in files:
                f = open(path + '/filter/' + name, 'r')
                bp = f.readlines()[0]
                name = name.strip('.txt')
                f.close()
                filter_dict[name] = bp
    except:
        print("檔案夾不存在。")

    # load files for scenario reproducing
    transform_dict = {}
    velocity_dict = {}
    for actor_id, _ in filter_dict.items():
        transform_dict[actor_id] = read_transform(os.path.join(path, 'transform', actor_id + '.npy'))
        velocity_dict[actor_id] = read_velocity(os.path.join(path, 'velocity', actor_id + '.npy'))
    num_files = len(filter_dict)

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0, 0, 0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
            
        weather = args.weather
        exec("args.weather = carla.WeatherParameters.%s" % args.weather)
        world = World(client.load_world(args.map), filter_dict['player'], hud, args)            
        client.get_world().set_weather(args.weather)     
        
        # sync mode                
        settings = world.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True # Enables synchronous mode
        world.world.apply_settings(settings)

        # other setting
        controller = KeyboardControl(world, args.autopilot)
        blueprint_library = client.get_world().get_blueprint_library()
        # lm = world.world.get_lightmanager()
        # lights = lm.get_all_lights()
        lights = []
        actors = world.world.get_actors()
        for l in actors:
            if 5 in l.semantic_tags and 18 in l.semantic_tags:
                lights.append(l)
        light_dict = read_traffic_lights(path, lights)

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

        # set controller
        for actor_id, bp in filter_dict.items():
            if actor_id != 'player':
                transform_spawn = transform_dict[actor_id][0]
                transform_spawn.location.z += 2
                agents_dict[actor_id] = client.get_world().spawn_actor(
                    set_bp(blueprint_library.filter(filter_dict[actor_id]), actor_id), 
                    transform_spawn)

            if 'vehicle' in bp:
                controller_dict[actor_id] = VehiclePIDController(agents_dict[actor_id], args_lateral={'K_P': 1, 'K_D': 0.0, 'K_I': 0}, args_longitudinal={'K_P': 1, 'K_D': 0.0, 'K_I': 0.0},
                                                            max_throttle=1.0, max_brake=1.0, max_steering=1.0)
            elif 'pedestrian' in bp:
                controller_dict[actor_id] = client.get_world().spawn_actor(
                                        blueprint_library.find('controller.ai.walker'),
                                        carla.Transform(), 
                                        attach_to=agents_dict[actor_id])
            
            actor_transform_index[actor_id] = 1
            finish[actor_id] = False

        waypoints = client.get_world().get_map().generate_waypoints(distance=1.0)

        # time.sleep(2)

        # dynamic scenario setting
        root = os.path.join('data_collection', args.scenario_id) 
        scenario_name = str(weather) + '_'
        if args.random_objects:
            t = threading.Thread(target = auto_spawn_object,args=(world, 5))
            t.start()
            scenario_name = scenario_name + 'random_objects_'

        if args.random_actors != 'None':
            if args.random_actors == 'low':
                spawn_actor_nearby(distance=100, vehicles=5, pedestrian=2, transform_dict=transform_dict)
            elif args.random_actors == 'mid':
                spawn_actor_nearby(distance=100, vehicles=10, pedestrian=5, transform_dict=transform_dict)
            elif args.random_actors == 'high':
                spawn_actor_nearby(distance=100, vehicles=20, pedestrian=10, transform_dict=transform_dict)
            scenario_name = scenario_name + args.random_actors + '_'
        
        # recording traj
        # id = []
        # moment = []
        # with open("data_collection/" + args.scenario_id + "/timestamp.txt") as f:
        #     for line in f.readlines():
        #         s = line.split(',')
        #         id.append(int(s[0]))
        #         moment.append(s[1])
        # period = float(moment[-1]) - float(moment[0])
        # half_period = period / 2
        # traj_col = threading.Thread(target = collect_trajectory,args=(world, world.player, args.scenario_id, period))
        # traj_col.start()

        # topo_col = threading.Thread(target = collect_topology,args=(world, world.player, args.scenario_id, half_period))
        # topo_col.start()

        # dynamic scenario setting
        # scenario_name = scenario_name + 'noise_trajectory' if args.noise_trajectory else scenario_name
        # stored_path = os.path.join(root, scenario_name)
        # print(stored_path)
        # if not os.path.exists(stored_path):
        #     os.makedirs(stored_path)

        scenario_finished = False
        iter_tick = 0
        iter_start = 25
        while (1):
            clock.tick_busy_loop(40)
            frame = world.world.tick()
            iter_tick += 1
            if iter_tick == iter_start + 1:
                pass
                # world.camera_manager.toggle_recording(stored_path) 
                # world.imu_sensor.toggle_recording_IMU(stored_path)
                # traj_col = threading.Thread(target = collect_trajectory,args=(world, world.player, args.scenario_id, period))
                # traj_col.start()

                # topo_col = threading.Thread(target = collect_topology,args=(world, world.player, args.scenario_id, half_period))
                # topo_col.start()
            elif iter_tick > iter_start:
                # iterate actors
                for actor_id, _ in filter_dict.items():
                    # apply recorded location and velocity on the controller

                    # reproduce traffic light state
                    if actor_id == 'player':
                        set_light_state(lights, light_dict, actor_transform_index[actor_id])
                    if actor_transform_index[actor_id] < len(transform_dict[actor_id]):
                        if 'vehicle' in filter_dict[actor_id]:
                            agents_dict[actor_id].apply_control(controller_dict[actor_id].run_step(
                                (velocity_dict[actor_id][actor_transform_index[actor_id]])*3600/1000.0, transform_dict[actor_id][actor_transform_index[actor_id]]))

                            v = agents_dict[actor_id].get_velocity()
                            v = ((v.x)**2 + (v.y)**2+(v.z)**2)**(0.5)


                            # to avoid the actor slowing down for the dense location around
                            # if agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) < 2 + v/20.0:
                            if agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) < 2.0:
                                if args.noise_trajectory:
                                    # sampling location with larger distance
                                    # actor_transform_index[actor_id] += max(1, int(7 + v//5.0))
                                    actor_transform_index[actor_id] += 8
                                else:
                                    # actor_transform_index[actor_id] += max(1, int(7 + v//10.0))
                                    actor_transform_index[actor_id] += 2
                            elif agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) > 6.0:
                                actor_transform_index[actor_id] += 6
                            else:
                                actor_transform_index[actor_id] += 1

                            # save ego-car's speed, control, transform
                            # if actor_id == 'player':
                            #     world.record_speed_control_transform(frame)

                        elif 'pedestrian' in filter_dict[actor_id]:
                            if actor_transform_index[actor_id] == 1:
                                # try:
                                #     controller_dict[actor_id].start()
                                # except:
                                #     print('sth wrong w/ walker_start')

                                # controller_dict[actor_id].go_to_location(transform_dict[actor_id][actor_transform_index[actor_id]].location)
                                controller_dict[actor_id].go_to_location(transform_dict[actor_id][-1].location)
                            # controller_dict[actor_id].set_max_speed(velocity_dict[actor_id][actor_transform_index[actor_id]])
                                controller_dict[actor_id].set_max_speed(1.4)
                            # if agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][actor_transform_index[actor_id]].location) < 1.5:
                            #     actor_transform_index[actor_id] += 10
                            # else:
                            #     actor_transform_index[actor_id] += 7
                    else:
                        # when the client has arrived the last recorded location
                        # if agents_dict[actor_id].get_transform().location.distance(transform_dict[actor_id][-1].location) > 3.0:
                        #     if 'vehicle' in filter_dict[actor_id]:
                        #         agents_dict[actor_id].apply_control(controller_dict[actor_id].run_step(
                        #             velocity_dict[actor_id][-30], transform_dict[actor_id][-1]))
                        #     elif 'pedestrian' in filter_dict[actor_id]:
                        #         controller_dict[actor_id].go_to_location(transform_dict[actor_id][-1].location)
                        #         controller_dict[actor_id].set_max_speed(1.4)

                        # else:
                        finish[actor_id] = True

                        # elif actor_id == 'player':
                        #     scenario_finished = True
                        #     break
                if not False in finish.values():
                        break

                if controller.parse_events(client, world, clock) == 1:
                    return

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            # if scenario_finished:
            #     break
        # end_frame = client.get_world().wait_for_tick().frame
        #world.save_speed_control_transform(stored_path)
        # world.imu_sensor.toggle_recording_IMU(stored_path)
        #world.collision_sensor.save_history(stored_path)
        # world.camera_manager.toggle_recording(stored_path)
        #save_description(world, args, stored_path, weather)
    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

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
        '-scenario_id',
        type=str,
        help='name of the scenario')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
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
        '-weather',
        default='ClearNoon',
        type=str,
        choices=['ClearNoon', 'CloudyNoon',
                'WetNoon', 'WetCloudyNoon',
                'MidRainyNoon', 'HardRainNoon'
                'SoftRainNoon', 'ClearSunset',
                'CloudySunset','WetSunset',
                'WetCloudySunset', 'MidRainSunset',
                'HardRainSunset', 'SoftRainSunset'],
        help='weather name')
    argparser.add_argument(
        '-map',
        default='Town03',
        type=str,
        help='map name')
    argparser.add_argument(
        '-random_actors',
        type=str,
        default='None',
        choices=['None', 'low', 'mid', 'high'],
        help='enable roaming actors')
    argparser.add_argument(
        '-random_objects',
        type=bool,
        default=False,
        help='enable random objects')
    argparser.add_argument(
        '-noise_trajectory',
        type=bool,
        default=False,
        help='apply noise on trajectory')

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
