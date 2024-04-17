import cv2
import numpy as np
import pandas as pd

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def track_sperms(video_path):
    cap = cv2.VideoCapture(video_path)
    _, first_frame = cap.read()
    gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Detect initial points to track (may need to adjust parameters for your video)
    initial_points = cv2.goodFeaturesToTrack(gray_first_frame, maxCorners=100, qualityLevel=0.3, minDistance=7)

    # Create a mask for drawing purposes
    mask = np.zeros_like(first_frame)

    tracks = {i: [point[0]] for i, point in enumerate(initial_points)}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(gray_first_frame, gray_frame, initial_points, None)

        for i, (new, old) in enumerate(zip(new_points, initial_points)):
            if status[i] == 1:
                a, b = new.ravel()
                c, d = old.ravel()
                tracks[i].append((a, b))
                mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)

        gray_first_frame = gray_frame.copy()
        initial_points = new_points

    cap.release()

    # Calculate distances
    distances = {track_id: sum(calculate_distance(tracks[track_id][i], tracks[track_id][i+1]) for i in range(len(tracks[track_id])-1)) for track_id in tracks}
    return distances

video_path = 'Video 8 without tracks.mp4'
distances = track_sperms(video_path)

# Saving distances to a CSV file
df = pd.DataFrame.from_dict(distances, orient='index', columns=['Distance'])
df.to_csv('sperm_tracking_results.csv')

print("Tracking completed. Results saved to sperm_tracking_results.csv")
