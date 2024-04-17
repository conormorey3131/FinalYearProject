import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from threading import Thread
from PIL import Image, ImageTk
import time

class VideoPlayerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Lab Video Player")
        
        # Layout configuration
        self.setup_layout()
        
        # Video playback threading
        self.playback_thread = None
        self.paused = False
        self.quit_playback = False

        # Playback variables
        self.cap1 = None
        self.cap2 = None

    def setup_layout(self):
        # Label for the first video path input
        self.label_video_path1 = tk.Label(self.master, text="Enter path for first video:")
        self.label_video_path1.pack()
        
        # Entry for the first video path
        self.video_path1_entry = tk.Entry(self.master)
        self.video_path1_entry.pack()
        
        # Label for the second video path input
        self.label_video_path2 = tk.Label(self.master, text="Enter path for second video:")
        self.label_video_path2.pack()
        
        # Entry for the second video path
        self.video_path2_entry = tk.Entry(self.master)
        self.video_path2_entry.pack()

        # Playback controls
        self.play_button = tk.Button(self.master, text="Play", command=self.start_playback)
        self.play_button.pack()
        
        self.pause_button = tk.Button(self.master, text="Pause", command=self.pause_playback)
        self.pause_button.pack()

        self.resume_button = tk.Button(self.master, text="Resume", command=self.resume_playback)
        self.resume_button.pack()
        
        self.quit_button = tk.Button(self.master, text="Quit", command=self.quit_playback_func)
        self.quit_button.pack()

        # Video display label
        self.video_display = tk.Label(self.master)
        self.video_display.pack()

    def start_playback(self):
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.quit_playback = False
            self.paused = False
            video_path1 = self.video_path1_entry.get()
            video_path2 = self.video_path2_entry.get()
            self.playback_thread = Thread(target=self.play_and_replay_videos_with_titles, args=(video_path1, video_path2), daemon=True)
            self.playback_thread.start()

    def pause_playback(self):
        self.paused = True

    def resume_playback(self):
        self.paused = False

    def quit_playback_func(self):
        self.quit_playback = True
        if self.cap1 is not None:
            self.cap1.release()
        if self.cap2 is not None:
            self.cap2.release()
        cv2.destroyAllWindows()
        self.master.quit()

    def play_and_replay_videos_with_titles(self, video_path1, video_path2, window_size=(640, 480), border_width=10):
        # Frame delay to achieve ~2 frames per second
        frame_delay = 1 / 2  # 0.5 seconds per frame

        self.cap1 = cv2.VideoCapture(video_path1)
        self.cap2 = cv2.VideoCapture(video_path2)

        # Create a black border
        border = np.zeros((window_size[1], border_width, 3), dtype=np.uint8)

        while not self.quit_playback:
            if not self.paused:
                ret1, frame1 = self.cap1.read()
                ret2, frame2 = self.cap2.read()

                if not ret1 or not ret2:
                    self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame1 = cv2.resize(frame1, (window_size[0]//2 - border_width//2, window_size[1]))
                frame2 = cv2.resize(frame2, (window_size[0]//2 - border_width//2, window_size[1]))

                combined_frame = np.hstack((frame1, border, frame2))
                
                # Convert to a format Tkinter can handle
                combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
                combined_image = Image.fromarray(combined_frame)
                combined_image_tk = ImageTk.PhotoImage(image=combined_image)
                
                # Display in Tkinter label
                self.video_display.configure(image=combined_image_tk)
                self.video_display.image = combined_image_tk

                time.sleep(frame_delay)  # Wait to maintain ~2 fps

            self.master.update_idletasks()
            self.master.update()

# Create the GUI application
root = tk.Tk()
app = VideoPlayerGUI(root)
root.mainloop()
