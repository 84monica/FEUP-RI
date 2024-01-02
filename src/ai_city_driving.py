import argparse
from PIL import Image
import gym
from gym_duckietown.envs import DuckietownEnv
import os
from roboflow import Roboflow

# Path to custom map
current_dir = os.path.dirname(os.path.abspath(__file__))
map_file = os.path.join(current_dir, 'map.yaml')

# Initiatize Model
rf = Roboflow(api_key="UUZRYYia676DD0a4CbI2")
project = rf.workspace().project("ri-h9xbe")
model = project.version(3).model

# Initialize DuckieTown Simulation
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default=map_file)
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name=args.map_name,
        domain_rand=False,
        draw_bbox=False,
    )
else:
    env = gym.make(args.env_name)

obs = env.reset()
env.render()

stop_sign = False
stop_steps = 0
robot_detected = False

while True:
    if robot_detected:
        # Stop until the robot is gone

        speed = 0
        steering = 0

    elif stop_sign:
        # Is at stop sign with no robots detected, stop for 30 steps

        speed = 0
        steering = 0
        
        stop_steps += 1
        
        if stop_steps > 30:
            stop_sign = False
            stop_steps = 0
            
    else:
        # Calculate the speed and direction.        
        
        lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        distance_to_road_center = lane_pose.dist
        angle_from_straight_in_rads = lane_pose.angle_rad

        k_p = 20
        k_d = 10

        # The speed is a value between [0, 1] (which corresponds to a real speed between 0m/s and 1.2m/s)
        speed = 0.3
    
        # Angle of the steering wheel, which corresponds to the angular velocity in rad/s
        steering = k_p*distance_to_road_center + k_d*angle_from_straight_in_rads 
    
    # Make step
    obs, reward, done, info = env.step([speed, steering])
    # print('Steps = %s, Timestep Reward=%.3f, ' % (env.step_count, reward))

    # Check for any objects
    if env.step_count % 20 == 0:
        print("Checking for objects...\n")

        # Convert the observation into an image
        im = Image.fromarray(obs)
        im.save("image.jpg")

        # Make a prediction based on the image
        predictions = model.predict("image.jpg", confidence=40, overlap=30).json()["predictions"]
        
        for prediction in predictions:

            # Extract type of object found in the image and confidence that it was found
            object_detected = prediction["class"]
            confidence = prediction["confidence"]
            
            # Extract coords and size of where the object was found in the image
            x, y, w, h = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
            
            # Print object detected info
            if (object_detected == "t_intersect"):
                print("Intersection detected! \n" + "(x=" + str(x) + ",y=" + str(y) + ",w=" + str(w) + ",h=" + str(h) + ")\t Confidence: " + str(confidence) + "\n")
            elif (object_detected == "stop"):
                print("Stop sign detected! \n" + "(x=" + str(x) + ",y=" + str(y) + ",w=" + str(w) + ",h=" + str(h) + ")\t Confidence: " + str(confidence) + "\n")
            elif (object_detected == "robot"):
                print("Robot detected! \n" + "(x=" + str(x) + ",y=" + str(y) + ",w=" + str(w) + ",h=" + str(h) + ")\t Confidence: " + str(confidence) + "\n")
            
            # Robot is near stop or t-intersection sign
            if (object_detected == "stop" or object_detected == "t_intersect") and confidence > 0.5 and x > 300 and w > 100:
                if (object_detected == "stop"):
                    print("STOP!!!!!!")
                    stop_sign = True
                else:
                    print("INTERSECTION!!!!!!")
                
                # Detected a robot in the intersection
                if any(pred["class"] == "robot" and pred["confidence"] > 0.5 and pred["width"] > 60 for pred in predictions):
                    robot_detected = True
                    print("ROBOT DETECTED!!!!!! \n")
                else:
                    print("NO ROBOT DETECTED!!!!!! \n")
                            
    # Render Simulation
    env.render()

    # In case it goes off the road or crashes
    if done:
        if reward < 0:
            print('*** CRASHED ***')
        else:
            print('*** OFF ROAD ***')     

        # Reset environment, respawn and continue simulation 
        env.reset()
        done = False