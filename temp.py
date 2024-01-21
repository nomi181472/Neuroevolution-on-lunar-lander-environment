import gym
from gym.wrappers.record_video import RecordVideo
from gym.wrappers.monitoring.video_recorder import VideoRecorder

# Create the CartPole environment
env = gym.make('LunarLander-v2',render_mode="human")
render_modes = env.metadata.get('render.modes')
if render_modes:
    print("Rendering modes:", render_modes)
else:
    print("Rendering is not supported for this environment.")

# Wrap the environment with Monitor to record videos
#env = RecordVideo(env, video_folder='/videos',episode_trigger=lambda x:x% 1==0)

# Create a VideoRecorder to record the video
video_recorder = VideoRecorder(env, path='videos/video.mp4', enabled=True)

# Reset the environment to get the initial observation
obs,_ = env.reset()

# Perform some steps in the environment
for step in range(200):
    # Take a random action (replace this with your agent's action)
    action = env.action_space.sample()

    # Perform the action in the environment
    obs, reward, done,termination, _ = env.step(action)

    # Print the reward
    print(f"Step: {step + 1}, Reward: {reward}")

    # Modify the rendering process to include the reward
    frame = env.render()
    #env.viewer.window.text('Reward: {:.2f}'.format(reward), 10, 10, size=12)

    # Record the frame for the video
    video_recorder.capture_frame()

    # Check if the episode is done
    if done:
        break

# Close the environment and save the video
env.close()
#video_recorder.close()
