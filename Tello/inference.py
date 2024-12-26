from crossformer.model.crossformer_model import CrossFormerModel
from djitellopy import Tello
import cv2
import numpy as np
import jax
import time

# Initialize and connect to the Tello drone
drone = Tello()
drone.connect()
drone.streamon()

# Load the pretrained CrossFormer model
start = time.time()
model = CrossFormerModel.load_pretrained("/home/jeanho/text_100000/text_100000")
unnorm = model.dataset_statistics["action"]

# Create navigation task
task = model.create_tasks(texts=["navigate forward"])
observation = {
    "image_nav": np.random.randint(0, 256, (1, 1, 224, 224, 3), dtype=np.uint8) / 255.0,
    "timestep_pad_mask": np.array([[True]]),
}
key = jax.random.PRNGKey(0)
action = model.sample_actions(observation, task, head_name="nav", unnormalization_statistics=unnorm, rng=key)
print('CrossFormer loaded in:', time.time() - start)

# Initialize video writer for recording
video_writer = None
frame_width, frame_height = 224, 224  # Match the resized frame dimensions
output_filename = "tello_navigation.avi"
video_writer = cv2.VideoWriter(
    output_filename, cv2.VideoWriter_fourcc(*'XVID'), 4, (frame_width, frame_height)
)

# Take off
to = False
try:
    frequency = 4  # 4Hz frequency
    delta_t = 1 / frequency  # Time interval in seconds (0.25 sec)

    while True:
        # Capture and preprocess the frame
        frame = drone.get_frame_read().frame
        frame = cv2.resize(frame, (224, 224))  # Resize to match model input
        frame_preserved = frame.copy()  # Preserve the original frame for display

        # Normalize and prepare the input frame
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension
        frame = np.expand_dims(frame, axis=0)  # Add time-step dimension

        # Prepare observation for the model
        observation = {
            "image_nav": frame,
            "timestep_pad_mask": np.array([[True]]),
        }

        # Generate action from the model
        drone.send_rc_control(0, 0, 0, 0)
        action = model.sample_actions(
            observation, task, head_name="nav",
            unnormalization_statistics=unnorm, rng=key
        )
        if not to:
            drone.takeoff()
            to = True
        # Extract waypoints (e.g., 4x2 matrix)
        waypoints = action[0]

        for waypoint in waypoints:
            original_frame = frame_preserved.copy()
            delta_x, delta_y = waypoint
            delta_x = int(delta_x)
            delta_y = int(delta_y)

            # Draw an arrow on the frame to visualize the action
            start_point = (224 // 2, 224 - 50)  # Arrow starts near the bottom center
            arrow_length = 1  # Length of the arrow

            # Calculate the endpoint of the arrow
            end_point = (
                int(start_point[0] + arrow_length * delta_y),
                int(start_point[1] - arrow_length * delta_x),
            )

            # Draw the arrow
            cv2.arrowedLine(
                original_frame, start_point, end_point,
                color=(0, 255, 0), thickness=1, tipLength=0.3
            )

            # Add text overlay for instructions and actions
            action_text = f"Action - Forward/Backward: {delta_x}, Yaw: {delta_y}"
            instruction_text = "Instruction: Navigate the hallway"

            cv2.putText(
                original_frame, action_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1
            )
            cv2.putText(
                original_frame, instruction_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1
            )

            # Write the frame to the video
            video_writer.write(original_frame)

            # Display the frame with the arrow and text
            cv2.imshow("Tello Camera", original_frame)

            # Maintain constant altitude and no roll
            throttle = 0
            roll = 0

            # Send command to the drone
            drone.send_rc_control(roll, int(delta_x), throttle, int(delta_y))

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Maintain 4Hz frequency
            time.sleep(delta_t)

except KeyboardInterrupt:
    print("Streaming stopped by user")

finally:
    # Safely land and stop the video stream
    drone.send_rc_control(0, 0, 0, 0)  # Stop any movement before landing
    drone.land()
    drone.streamoff()
    if video_writer:
        video_writer.release()  # Release the video writer
    cv2.destroyAllWindows()
