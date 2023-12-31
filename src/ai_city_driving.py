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
model = project.version(2).model

# Initialize DuckieTown Simulation
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default=map_file)
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
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

while True:
    if stop_sign:
        # Is at stop sign so stop for 30 steps

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

    # Check for any signs
    if env.step_count % 20 == 0:
        # print("Checking for signs...\n")

        # Convert the observation into an image
        im = Image.fromarray(obs)
        im.save("image.jpg")

        # Make a prediction based on the image
        predictions = model.predict("image.jpg", confidence=40, overlap=30).json()["predictions"]
        
        if (predictions != []):
            # Extract type of sign found in the image and confidence that it was found
            type_of_sign = predictions[0]["class"]
            confidence = predictions[0]["confidence"]
            
            # Extract coords and size of where the sign was found in the image
            x, y, w, h = predictions[0]["x"], predictions[0]["y"], predictions[0]["width"], predictions[0]["height"]
            
            # Print the sign info
            if (type_of_sign == "t_intersect"):
                print("Intersection ahead! \n" + "X = " + str(x) + ", Y = " + str(y) + ", W = " + str(w) + ", H =" + str(h) + "\n Confidence: " + str(confidence) + "\n")
            elif (type_of_sign == "stop"):
                print("Stop sign ahead! \n" + "X = " + str(x) + ", Y = " + str(y) + ", W = " + str(w) + ", H =" + str(h) + "\n Confidence: " + str(confidence) + "\n")
            
            # If the sign is a stop sign, the confidence is high enough and 
            # the stop sign is at the right side of the screen and is big on the screen then stop the car
            if (confidence > 0.5):
                if (type_of_sign == "stop"):
                    # Sign is at right of the screen
                    if (x > 300):
                        # Sign is bigger than 100 pixels
                        if (w > 100):
                            print("STOP!!!!!! \n")
                            stop_sign = True
            
            
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