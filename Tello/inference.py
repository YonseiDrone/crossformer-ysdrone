from crossformer.model.crossformer_model import CrossFormerModel
from djitellopy import Tello
import cv2
import numpy as np
import jax
import time
import threading

# Initialize and connect to the Tello drone
# Start the video stream
# drone.takeoff()   # Start flying

drone = Tello()
drone.connect()
drone.streamon()  
# Thread function to keep the drone active
actions_ready = False  # Flag to indicate when the action is ready

# def keep_drone_active():
#     while not actions_ready:
#         drone.send_keepalive()
#         time.sleep(1)  # Send this command every second
# thread = threading.Thread(target=keep_drone_active)
# thread.start()
# # Start the thread to keep the drone active

# Load the pre-trained model
start = time.time()
model = CrossFormerModel.load_pretrained("hf://rail-berkeley/crossformer")

# Create navigation task
task = model.create_tasks(texts=["navigate forward"])
observation = {
    "image_nav": np.random.randint(0, 256, (1, 1, 224, 224, 3), dtype=np.uint8) / 255.0,
    "timestep_pad_mask": np.array([[True]]),
}
action = model.sample_actions(observation, task, head_name="nav", rng=jax.random.PRNGKey(0))
print('crossformer loaded', time.time() - start)
start = time.time()
drone.takeoff()

try:
    frequency = 4  # 4Hz frequency
    delta_t = 1 / frequency  # Time interval in seconds (0.25 sec)

    while True:
        # Capture and preprocess the frame
        frame = drone.get_frame_read().frame
        frame = cv2.resize(frame, (224, 224))  # Resize to match model input
        # frame = frame / 255.0  # Normalize pixel values
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension
        frame = np.expand_dims(frame, axis=0)  # Add time-step dimension

        # Prepare observation for the model
        observation = {
            "image_nav": frame,
            "timestep_pad_mask": np.array([[True]]),
        }

        # Generate action from the model
        drone.send_rc_control(0, 0, 0, 0)
        action = model.sample_actions(observation, task, head_name="nav", rng=jax.random.PRNGKey(0))
        waypoints = action[0]  # Extract waypoints (e.g., 4x2 matrix)

        # Process waypoints at 4Hz
        for waypoint in waypoints:
            delta_x, delta_y = waypoint

            # Map delta_x to forward/backward speed
            alpha = 10.0
            forward_speed = int(max(-alpha, min(alpha, int(delta_x * alpha))))  # Scale to range -100 to 100

            # Map delta_y to yaw rate
            yaw_rate = int(max(-alpha, min(alpha, int(delta_y * alpha))))  # Scale to range -100 to 100

            # Maintain constant altitude and no roll
            throttle = 0
            roll = 0

            # Send command to the drone
            drone.send_rc_control(roll, forward_speed, throttle, yaw_rate)

            # Maintain 4Hz frequency
            time.sleep(delta_t)

        # Display the live feed (optional)
        cv2.imshow("Tello Camera", frame[0, 0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit loop on 'q' key press

except KeyboardInterrupt:
    print("Streaming stopped by user")

finally:
    # Safely land and stop the video stream
    actions_ready = True  # Signal thread to stop
    drone.send_rc_control(0, 0, 0, 0)  # Stop any movement before landing
    drone.land()
    drone.streamoff()
    cv2.destroyAllWindows()
