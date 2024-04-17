# Import necessary libraries
import cv2
import numpy as np
import pandas as pd
import random

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    # Use numpy to calculate the norm (distance) between the points
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to calculate speed
def calculate_speed(distance, total_frames, fps):
    # Calculate time in seconds as total frames divided by FPS
    time_seconds = total_frames / fps
    # Return speed as distance divided by time in seconds
    return distance / time_seconds

# Function to generate a random color
def generate_random_color():
    # Return a tuple with three elements (R, G, B), each a random integer between 0 and 255
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Main function to track sperms and generate video with their paths
def track_sperms_and_generate_video(video_path, output_video_path):
    # Open the input video file
    cap = cv2.VideoCapture(video_path)
    # Get FPS and total frames of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the first frame and convert it to grayscale
    _, first_frame = cap.read()
    gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Detect initial points of interest to track
    initial_points = cv2.goodFeaturesToTrack(gray_first_frame, maxCorners=150, qualityLevel=0.2, minDistance=5)

    # Setup video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    # Initialize track image and tracks dictionary
    track_image = np.zeros_like(first_frame)
    tracks = {i: [point[0]] for i, point in enumerate(initial_points)}
    colors = [generate_random_color() for _ in range(len(initial_points))]

    # Loop through each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate new positions using optical flow
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(gray_first_frame, gray_frame, initial_points, None)

        # Update tracks and draw paths
        for i, (new, old) in enumerate(zip(new_points, initial_points)):
            if status[i] == 1:
                a, b = new.ravel()
                c, d = old.ravel()
                tracks[i].append((a, b))
                color = colors[i]
                cv2.line(track_image, (int(a), int(b)), (int(c), int(d)), color, 2)

        # Add track image to current frame
        frame_with_tracks = cv2.add(frame, track_image)

        # Update for next iteration
        gray_first_frame = gray_frame.copy()
        initial_points = new_points

        # Write updated frame to output video
        out.write(frame_with_tracks)

    # Release video objects
    cap.release()
    out.release()

    # Calculate total distance moved and speed for each track
    distances = {track_id: sum(calculate_distance(tracks[track_id][i], tracks[track_id][i+1]) for i in range(len(tracks[track_id])-1)) for track_id in tracks}
    speeds = {track_id: calculate_speed(distance, total_frames, fps) for track_id, distance in distances.items()}
    sorted_speeds = dict(sorted(speeds.items(), key=lambda item: item[1], reverse=True))

    # Identify fastest sperms
    num_fastest = 5  # Number of fastest sperms to highlight
    fastest_sperms = list(sorted_speeds.keys())[:num_fastest]

    return distances, speeds, fastest_sperms

# User input for video paths
video_path = input("Enter the path of the video file: ")
output_video_path = 'Tracked_Sperms.avi'

# Execute main function
distances, speeds, fastest_sperms = track_sperms_and_generate_video(video_path, output_video_path)

# Saving results to a CSV file
df = pd.DataFrame.from_dict({
    'Distance': distances, 
    'Speed': speeds
})
df.to_csv('sperm_tracking_results.csv')

# Display completion message and IDs of fastest sperms
print(f"Tracking completed. Results saved to sperm_tracking_results.csv and video saved to Tracked_Sperms.avi")
print(f"The IDs of the fastest moving sperms are: {fastest_sperms}")
