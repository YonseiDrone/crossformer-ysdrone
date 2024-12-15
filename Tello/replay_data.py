import h5py
from djitellopy import Tello
import time

# Initialize Tello drone
def init_tello():
    """
    Initialize the Tello drone for replay.
    """
    tello = Tello()
    tello.connect()
    tello.streamon()  # Optional: Stream for live feed during replay
    tello.takeoff()
    print("Drone ready for replay.")
    return tello

# Replay flight: control the drone
def replay_flight(file_name, tello):
    """
    Replay the saved flight data by controlling the drone.
    """
    with h5py.File(file_name, "r") as hdf5_file:
        actions = hdf5_file["action"]
        li = hdf5_file['language_instruction']

        print(f"Replaying {len(actions)} actions...")

        loop_interval = 0.25  # 4Hz replay rate

        try:
            for i in range(len(actions)):
                loop_start_time = time.time()

                # Retrieve saved control actions
                forward_backward, yaw = actions[i]
                print(li[i])

                # Send the saved commands to the drone
                tello.send_rc_control(0, int(forward_backward), 0, int(yaw))

                # Maintain the replay interval
                elapsed_time = time.time() - loop_start_time
                if elapsed_time < loop_interval:
                    time.sleep(loop_interval - elapsed_time)

        except Exception as e:
            print(f"Error during replay: {e}")
        finally:
            #tello.send_rc_control(0, 0, 0, 0)  # Stop the drone after replay
            print("Replay complete.")

if __name__ == "__main__":
    # Load saved flight data and replay
    tello = init_tello()
    try:
        replay_flight("tello_flight_data_1_n.h5", tello)
    finally:
        # Ensure the drone lands and resources are cleaned up
        tello.land()
        tello.end()
