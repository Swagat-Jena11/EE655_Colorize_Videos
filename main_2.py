import os
import subprocess
import time
import cv2
import numpy as np

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIR = os.path.join(BASE_DIR, "videos", "source")
RESULT_DIR = os.path.join(BASE_DIR, "videos", "result")
TEMP_DIR = os.path.join(BASE_DIR, "videos", "temp")

REAL_ESRGAN_DIR = os.path.join(BASE_DIR, "Real-ESRGAN")
DEOLDIFY_DIR = os.path.join(BASE_DIR, "DeOldify")
NVIDIA_VFX_DIR = os.path.join(BASE_DIR, "nvidia-vfx-python-samples")

REAL_ESRGAN_PYTHON = os.path.join(
    REAL_ESRGAN_DIR,
    "realesr_env",
    "Scripts",
    "python.exe"
)

DEOLDIFY_PYTHON = os.path.join(
    DEOLDIFY_DIR,
    "deoldify_env",
    "Scripts",
    "python.exe"
)

REAL_ESRGAN_SCRIPT = os.path.join(
    REAL_ESRGAN_DIR,
    "inference_realesrgan_video.py"
)

NVIDIA_VFX_SCRIPT = os.path.join(
    NVIDIA_VFX_DIR,
    "video_super_resolution.py"
)

DEOLDIFY_SCRIPT = os.path.join(
    DEOLDIFY_DIR,
    "deoldify_noraft.py"
)

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
REAL_ESRGAN_MODEL = "realesr-general-x4v3"
REAL_ESRGAN_SCALE = 2
REAL_ESRGAN_DENOISE = 0.2

DEOLDIFY_RENDER_FACTOR = 21

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def run_command(cmd, cwd=None):
    print("\n" + "=" * 80)
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 80)

    start = time.time()
    subprocess.run(cmd, check=True, cwd=cwd)

    print(f"Finished in {format_time(time.time() - start)}")


def apply_flicker_correction(input_video, output_video):
    print("\nApplying flicker correction...")

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        raise Exception(f"Could not open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    ret, prev_frame = cap.read()

    if not ret:
        raise Exception("Could not read first frame")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_mean = np.mean(prev_gray)

    out.write(prev_frame)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_mean = np.mean(gray)

        scale = prev_mean / (curr_mean + 1e-6)
        scale = np.clip(scale, 0.90, 1.10)

        corrected = np.clip(
            frame.astype(np.float32) * scale,
            0,
            255
        ).astype(np.uint8)

        out.write(corrected)

        corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        corrected_mean = np.mean(corrected_gray)

        prev_mean = 0.95 * prev_mean + 0.05 * corrected_mean

    cap.release()
    out.release()

    print(f"Flicker-corrected video saved: {output_video}")


# -------------------------------------------------
# PROCESS ONE VIDEO
# -------------------------------------------------
def process_video(input_video):
    base_name = os.path.splitext(os.path.basename(input_video))[0]

    denoise_video = os.path.join(TEMP_DIR, f"{base_name}_denoise.mp4")
    deblur_video = os.path.join(TEMP_DIR, f"{base_name}_deblur.mp4")
    sr_video = os.path.join(TEMP_DIR, f"{base_name}_deblur_sr.mp4")
    color_video = os.path.join(TEMP_DIR, f"{base_name}_color.mp4")
    flicker_video = os.path.join(TEMP_DIR, f"{base_name}_flicker.mp4")
    final_video = os.path.join(RESULT_DIR, f"{base_name}_final.mp4")

    print("\n" + "#" * 80)
    print(f"Processing: {input_video}")
    print("#" * 80)

    # STEP 1: NVIDIA VFX DENOISE
    run_command([
        REAL_ESRGAN_PYTHON,
        NVIDIA_VFX_SCRIPT,
        "-i", input_video,
        "-o", denoise_video,
        "--scale", "1",
        "--quality", "DENOISE_ULTRA"
    ], cwd=NVIDIA_VFX_DIR)

    # STEP 2: NVIDIA VFX DEBLUR
    run_command([
        REAL_ESRGAN_PYTHON,
        NVIDIA_VFX_SCRIPT,
        "-i", denoise_video,
        "-o", deblur_video,
        "--scale", "1",
        "--quality", "DEBLUR_MEDIUM"
    ], cwd=NVIDIA_VFX_DIR)

    # STEP 3: REAL-ESRGAN FIRST PASS
    run_command([
        REAL_ESRGAN_PYTHON,
        REAL_ESRGAN_SCRIPT,
        "-i", deblur_video,
        "-n", REAL_ESRGAN_MODEL,
        "-o", TEMP_DIR,
        "--suffix", "sr",
        "-s", str(REAL_ESRGAN_SCALE),
        "-dn", str(REAL_ESRGAN_DENOISE)
    ], cwd=REAL_ESRGAN_DIR)

    # STEP 4: DEOLDIFY
    run_command([
        DEOLDIFY_PYTHON,
        DEOLDIFY_SCRIPT,
        sr_video,
        color_video,
        str(DEOLDIFY_RENDER_FACTOR)
    ], cwd=DEOLDIFY_DIR)

    # STEP 5: FLICKER CORRECTION
    apply_flicker_correction(color_video, flicker_video)

    # STEP 6: SAVE FINAL OUTPUT
    if os.path.exists(final_video):
        os.remove(final_video)

    os.replace(flicker_video, final_video)

    print(f"Final output saved: {final_video}")
    print(f"\nFinished processing: {base_name}")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    videos = [
        os.path.join(SOURCE_DIR, f)
        for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not videos:
        print("No videos found in videos/source")
        return

    total_start = time.time()

    for video in videos:
        try:
            process_video(video)
        except Exception as e:
            print(f"Failed for {video}: {e}")

    print("\n" + "=" * 80)
    print(f"ALL DONE in {format_time(time.time() - total_start)}")
    print("=" * 80)


if __name__ == "__main__":
    main()