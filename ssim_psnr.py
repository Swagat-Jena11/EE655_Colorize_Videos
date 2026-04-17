import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_video_metrics(video1_path, video2_path):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened():
        print(f"Could not open: {video1_path}")
        return

    if not cap2.isOpened():
        print(f"Could not open: {video2_path}")
        return

    psnr_values = []
    ssim_values = []

    frame_count = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Resize second frame to match first frame size
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Compute PSNR
        psnr_score = cv2.PSNR(frame1, frame2)

        # Compute SSIM
        ssim_score = ssim(gray1, gray2)

        psnr_values.append(psnr_score)
        ssim_values.append(ssim_score)

        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap1.release()
    cap2.release()

    if len(psnr_values) == 0 or len(ssim_values) == 0:
        print("No frames compared.")
        return

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    min_psnr = np.min(psnr_values)
    max_psnr = np.max(psnr_values)

    min_ssim = np.min(ssim_values)
    max_ssim = np.max(ssim_values)

    print("\n==============================")
    print(f"Frames compared : {frame_count}")
    print(f"Average PSNR    : {avg_psnr:.2f} dB")
    print(f"Minimum PSNR    : {min_psnr:.2f} dB")
    print(f"Maximum PSNR    : {max_psnr:.2f} dB")
    print()
    print(f"Average SSIM    : {avg_ssim:.4f}")
    print(f"Minimum SSIM    : {min_ssim:.4f}")
    print(f"Maximum SSIM    : {max_ssim:.4f}")
    print("==============================")

    return {
        "avg_psnr": avg_psnr,
        "min_psnr": min_psnr,
        "max_psnr": max_psnr,
        "avg_ssim": avg_ssim,
        "min_ssim": min_ssim,
        "max_ssim": max_ssim
    }


if __name__ == "__main__":
    video1 = "videos/source/DDay_Omaha_Eastern.mp4"
    video2 = "videos/result/DDay_Omaha_Eastern_color_final.mp4"

    calculate_video_metrics(video1, video2)