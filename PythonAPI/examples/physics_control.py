import carla
import random

def main():
    # Connect to client
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)

    # Get World and Actors
    world = client.get_world()
    current_map = world.get_map()
    actors = world.get_actors()

    # Get a random vehicle from world (there should be one at least)
    vehicle = random.choice([actor for actor in actors if 'vehicle' in actor.type_id])

    # Create Wheels Physics Control
    front_left_wheel  = carla.WheelPhysicsControl(tire_friction=4.5, damping_rate=1.0, max_steer_angle=70.0, radius=30.0)
    front_right_wheel = carla.WheelPhysicsControl(tire_friction=2.5, damping_rate=1.5, max_steer_angle=70.0, radius=25.0)
    rear_left_wheel   = carla.WheelPhysicsControl(tire_friction=1.0, damping_rate=0.2, max_steer_angle=0.0,  radius=15.0)
    rear_right_wheel  = carla.WheelPhysicsControl(tire_friction=1.5, damping_rate=1.3, max_steer_angle=0.0,  radius=20.0)

    wheels = [front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel]

    # Change Vehicle Physics Control parameters of the vehicle
    physics_control = vehicle.get_physics_control()

    physics_control.torque_curve = [carla.Vector2D(x=0, y=400), carla.Vector2D(x=1300, y=600)] #引擎轉速
    physics_control.max_rpm = 10000 #每分鐘轉速
    physics_control.moi = 1.0 #引擎轉動慣量
    physics_control.damping_rate_full_throttle = 0.0 #油門全開的阻尼比
    physics_control.use_gear_autobox = True #是否有自動變速箱
    physics_control.gear_switch_time = 0.5 #換檔速度
    physics_control.clutch_strength = 10 #離合器強度
    physics_control.mass = 10000 #質量
    physics_control.drag_coefficient = 0.25 #阻力係數
    physics_control.steering_curve = [carla.Vector2D(x=0, y=1), carla.Vector2D(x=100, y=1), carla.Vector2D(x=300, y=1)] #給定速度的最大轉向
    physics_control.wheels = wheels #車輪
    physics_control.use_sweep_wheel_collision = True #車輪是否可以空轉

    # Apply Vehicle Physics Control for the vehicle
    vehicle.apply_physics_control(physics_control)

if __name__ == '__main__':
    main()