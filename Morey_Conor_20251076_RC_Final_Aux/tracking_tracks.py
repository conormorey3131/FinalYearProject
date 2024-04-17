import cv2
import numpy as np
import pandas as pd

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def track_sperms_and_generate_video(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    _, first_frame = cap.read()
    gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Improved feature detection
    initial_points = cv2.goodFeaturesToTrack(gray_first_frame, maxCorners=150, qualityLevel=0.2, minDistance=5)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 1.0, (int(cap.get(3)), int(cap.get(4))))

    # Create a blank image for drawing tracks
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
                # Draw a line on the track image
                cv2.line(track_image, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)

        # Add the track image to the current frame
        frame_with_tracks = cv2.add(frame, track_image)

        gray_first_frame = gray_frame.copy()
        initial_points = new_points

        # Write the frame with tracks
        out.write(frame_with_tracks)

    cap.release()
    out.release()

    # Calculate distances
    distances = {track_id: sum(calculate_distance(tracks[track_id][i], tracks[track_id][i+1]) for i in range(len(tracks[track_id])-1)) for track_id in tracks}
    return distances

video_path = 'Video 8 without tracks.mp4'
output_video_path = 'Tracked_Sperms.avi'
distances = track_sperms_and_generate_video(video_path, output_video_path)

# Saving distances to a CSV file
df = pd.DataFrame.from_dict(distances, orient='index', columns=['Distance'])
df.to_csv('sperm_tracking_results.csv')

print("Tracking completed. Results saved to sperm_tracking_results.csv and video saved to Tracked_Sperms.avi")