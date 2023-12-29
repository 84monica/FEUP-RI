import argparse
from PIL import Image
import gym
from gym_duckietown.envs import DuckietownEnv
import os
from roboflow import Roboflow


current_dir = os.path.dirname(os.path.abspath(__file__))
map_file = os.path.join(current_dir, 'map.yaml')

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default=map_file)
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
args = parser.parse_args()

rf = Roboflow(api_key="UUZRYYia676DD0a4CbI2")
project = rf.workspace().project("ri-h9xbe")
model = project.version(2).model

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

total_reward = 0

while True:

    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rads = lane_pose.angle_rad

    # Calculate the speed and direction.

    k_p = 20
    k_d = 10
    
    # The speed is a value between [0, 1] (which corresponds to a real speed between 0m/s and 1.2m/s)
    speed = 0.3
    
    # Angle of the steering wheel, which corresponds to the angular velocity in rad/s
    steering = k_p*distance_to_road_center + k_d*angle_from_straight_in_rads 
    
    obs, reward, done, info = env.step([speed, steering])
    total_reward += reward

    # print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))

    # Check for any signs every 50 steps
    
    if env.step_count % 50 == 0:
        print("Checking for signs...")

        # Convert the observation into an image
        im = Image.fromarray(obs)
        im.save("image.jpg")
        
        # [{'x': 50.5, 'y': 130.0, 'width': 29.0, 'height': 46.0, 
        # 'confidence': 0.5285166501998901, 'class': 't_intersect', 
        # 'class_confidence': None, 'class_id': 1, 'tracker_id': None, 
        # 'image_path': 'image.jpg', 'prediction_type': 'ObjectDetectionModel'}]

        predictions = model.predict("image.jpg", confidence=40, overlap=30).json()["predictions"]
        
        if (predictions != []):
            type_of_sign = predictions[0]["class"]
            distance_to_intersection = predictions[0]["x"]
            confidence = predictions[0]["confidence"]
            
            if (type_of_sign == "t_intersect"):
                print("Intersection ahead! \n Distance: " + str(distance_to_intersection) + "\n Confidence: " + str(confidence))
            elif (type_of_sign == "stop"):
                print("Stop sign ahead! \n Distance: " + str(distance_to_intersection) + "\n Confidence: " + str(confidence))
            
    env.render()

    if done:
        if reward < 0:
            print('*** CRASHED ***')
        print ('Final Reward = %.3f' % total_reward)
        env.reset()
        done = False