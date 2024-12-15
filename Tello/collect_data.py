import os
import pygame
import h5py
from djitellopy import Tello
import cv2
import time
import numpy as np

# Pygame initialization
pygame.init()
WINDOW_SIZE = (400, 300)
pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Tello Drone Controller")

# Determine new file name with user input appended at the end
def get_new_file_name(base_name, user_input, extension):
    counter = 1
    while True:
        file_name = f"{base_name}_{counter}_{user_input}{extension}"
        if not os.path.exists(file_name):
            return file_name
        counter += 1

IMG_SHAPE = (360, 480, 3)  # Camera frame shape
ACTION_SHAPE = (2,)  # forward_backward, yaw

# Initialize HDF5
def init_hdf5(file_name):
    hdf5_file = h5py.File(file_name, "w")
    observation_dataset = hdf5_file.create_dataset(
        "observation", (0, *IMG_SHAPE), maxshape=(None, *IMG_SHAPE), dtype=np.uint8
    )
    action_dataset = hdf5_file.create_dataset(
        "action", (0, *ACTION_SHAPE), maxshape=(None, *ACTION_SHAPE), dtype=np.float32
    )
    language_dataset = hdf5_file.create_dataset(
        "language_instruction",
        (0,),  # Initialize with an empty dataset
        maxshape=(None,),  # Allow the dataset to grow in the first dimension
        dtype=h5py.string_dtype(encoding='utf-8')  # Use string data type with UTF-8 encoding
    )

    return hdf5_file, observation_dataset, action_dataset, language_dataset

# Initialize Tello drone
def init_tello():
    tello = Tello()
    tello.connect()
    tello.streamon()
    tello.takeoff()
    print("Drone connected and ready.")
    return tello

# Handle user control
def control_drone(keys, x_speed, yaw_speed):
    forward_backward = 0
    yaw = 0

    if keys[pygame.K_w]:  # Forward
        forward_backward = x_speed
    elif keys[pygame.K_s]:  # Backward
        forward_backward = -x_speed

    if keys[pygame.K_a]:  # Counter-clockwise rotation
        yaw = -yaw_speed
    elif keys[pygame.K_d]:  # Clockwise rotation
        yaw = yaw_speed

    return forward_backward, yaw

# Main loop for saving
def save_flight():
    # Get user input
    li = 'navigate the hallway'
    # li = 'make a u-turn'
    user_input = input("Enter a string to append to the filename: ")

    # Determine file name
    global FILE_NAME
    FILE_NAME = get_new_file_name("tello_flight_data", user_input, ".h5")

    tello = init_tello()
    hdf5_file, observation_dataset, action_dataset, language_dataset = init_hdf5(FILE_NAME)

    x_speed = 45 # Speed in cm/s
    yaw_speed = 40  # Speed in degrees/s
    loop_interval = 0.25  # Loop interval in seconds (4Hz)

    try:
        while True:
            loop_start_time = time.time()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            # Get keyboard inputs
            keys = pygame.key.get_pressed()
            forward_backward, yaw = control_drone(keys, x_speed, yaw_speed)

            # Send command to the drone
            tello.send_rc_control(0, forward_backward, 0, yaw)

            # Get and resize the camera frame
            frame = tello.get_frame_read().frame
            if frame is not None:
                resized_frame = cv2.resize(frame, IMG_SHAPE[:2][::-1])

                # Save observation and action
                observation_dataset.resize(observation_dataset.shape[0] + 1, axis=0)
                observation_dataset[-1] = resized_frame

                action_dataset.resize(action_dataset.shape[0] + 1, axis=0)
                action_dataset[-1] = [forward_backward, yaw]

                language_dataset.resize(language_dataset.shape[0] + 1, axis=0)
                language_dataset[-1] = li

                # Display the frame
                cv2.imshow("Tello Camera", resized_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

            # Maintain loop interval
            elapsed_time = time.time() - loop_start_time
            if elapsed_time < loop_interval:
                time.sleep(loop_interval - elapsed_time)

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        hdf5_file.close()
        tello.streamoff()
        tello.end()
        pygame.quit()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    save_flight()