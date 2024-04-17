import cv2
import numpy as np
import pandas as pd

# Function to detect the object and return its center position
def detect_object(frame, lower_color, upper_color):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Create a mask for color range
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour and its center
    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        return (int(x), int(y)), radius
    return None, 0

# Define color range for detection
lower_color = np.array([110, 50, 50])  # Example: lower blue
upper_color = np.array([130, 255, 255])  # Example: upper blue

# Initialize previous position
prev_center = None

# Initialize a list to store distance readings
distances = []

# Open the video capture
cap = cv2.VideoCapture('ball_bouncing.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect the object
    center, radius = detect_object(frame, lower_color, upper_color)

    # Draw and calculate movement
    if center and radius > 10:
        cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
        if prev_center:
            # Draw line between previous and current positions
            cv2.line(frame, prev_center, center, (0, 0, 255), 2)

            # Calculate movement
            distance = np.linalg.norm(np.array(prev_center) - np.array(center))
            print("Movement distance:", distance)

            # Append the distance to the list
            distances.append(distance)

        prev_center = center

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

# Convert the list of distances into a pandas DataFrame
df = pd.DataFrame(distances, columns=['Distance'])

# Save the DataFrame to an Excel file
df.to_excel('distance_readings.xlsx', index=False)
