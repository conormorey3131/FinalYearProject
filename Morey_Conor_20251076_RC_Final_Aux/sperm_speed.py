import cv2
import numpy as np
import pandas as pd

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_speed(distance, total_frames, fps):
    time_seconds = total_frames / fps
    return distance / time_seconds  # speed = distance / time

def track_sperms_and_generate_video(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    _, first_frame = cap.read()
    gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    initial_points = cv2.goodFeaturesToTrack(gray_first_frame, maxCorners=150, qualityLevel=0.2, minDistance=5)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    track_image = np.zeros_like(first_frame)

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
                cv2.line(track_image, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)

        frame_with_tracks = cv2.add(frame, track_image)

        gray_first_frame = gray_frame.copy()
        initial_points = new_points

        out.write(frame_with_tracks)

    cap.release()
    out.release()

    distances = {track_id: sum(calculate_distance(tracks[track_id][i], tracks[track_id][i+1]) for i in range(len(tracks[track_id])-1)) for track_id in tracks}
    speeds = {track_id: calculate_speed(distance, total_frames, fps) for track_id, distance in distances.items()}
    sorted_speeds = dict(sorted(speeds.items(), key=lambda item: item[1], reverse=True))

    return distances, sorted_speeds

# User input for video paths
video_path = input("Enter the path of the video file: ")
output_video_path = 'Tracked_Sperms.avi'

distances, sorted_speeds = track_sperms_and_generate_video(video_path, output_video_path)

# Saving results to a CSV file
df = pd.DataFrame.from_dict({
    'Distance': distances, 
    'Speed': sorted_speeds
})
df.to_csv('sperm_tracking_results.csv')

print("Tracking completed. Results saved to sperm_tracking_results.csv and video saved to Tracked_Sperms.avi")