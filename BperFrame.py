import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_frame_brightness_range(video_path, start_frame=10, end_frame=20):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return [], []

    brightness_values = []
    frame_indices = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if start_frame <= frame_idx <= end_frame:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)

            brightness_values.append(brightness)
            frame_indices.append(frame_idx)

        if frame_idx > end_frame:
            break

        frame_idx += 1

    cap.release()
    return frame_indices, brightness_values


video1 = "videos/temp/Indian Village And Market (1934) [Ydiz1Hzfx5s]_color.mp4"
video2 = "videos/result/Indian Village And Market (1934) [Ydiz1Hzfx5s]_final.mp4"

indices1, brightness1 = get_frame_brightness_range(
    video1,
    start_frame=10,
    end_frame=20
)

indices2, brightness2 = get_frame_brightness_range(
    video2,
    start_frame=10,
    end_frame=20
)

min_len = min(len(brightness1), len(brightness2))

brightness1 = brightness1[:min_len]
brightness2 = brightness2[:min_len]
frame_indices = indices1[:min_len]

x = np.arange(len(frame_indices))
bar_width = 0.35

plt.figure(figsize=(8, 6))

plt.bar(
    x - bar_width / 2,
    brightness1,
    width=bar_width,
    label="Original Video"
)

plt.bar(
    x + bar_width / 2,
    brightness2,
    width=bar_width,
    label="Processed Video"
)

plt.xticks(x, frame_indices)

plt.xlabel("Frame Index")
plt.ylabel("Average Brightness")
plt.title("Brightness Comparison for Frames 10 to 20")
plt.legend()
plt.tight_layout()
plt.show()