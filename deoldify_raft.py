import os
import cv2
import torch
import shutil
import numpy as np
from argparse import Namespace

# -----------------------------
# DeOldify setup
# -----------------------------
from deoldify import device
from deoldify.device_id import DeviceId

device.set(device=DeviceId.GPU0)

from deoldify.visualize import get_image_colorizer

# -----------------------------
# RAFT setup
# -----------------------------
import sys
sys.path.append("RAFT/core")

from raft import RAFT
from utils.utils import InputPadder

DEVICE = "cuda"

args = Namespace(
    small=False,
    mixed_precision=True,
    alternate_corr=False,
    dropout=0,
)

print("🔧 Loading RAFT model...")

raft_model = RAFT(args)

state_dict = torch.load(
    "RAFT/models/raft-things.pth",
    map_location=DEVICE
)

state_dict = {
    k.replace("module.", ""): v
    for k, v in state_dict.items()
}

raft_model.load_state_dict(state_dict)
raft_model = raft_model.to(DEVICE).eval().half()

print("✅ RAFT model loaded")

# -----------------------------
# DeOldify setup
# -----------------------------
colorizer = get_image_colorizer(artistic=True)

# -----------------------------
# Helper functions
# -----------------------------
def resize_frame(img, max_size=512):
    h, w = img.shape[:2]

    scale = max_size / max(h, w)

    if scale < 1:
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    return img


def load_image_as_tensor(path, max_size=512):
    img = cv2.imread(path)
    img = resize_frame(img, max_size=max_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(img).permute(2, 0, 1).float()[None]
    tensor = tensor.to(DEVICE).half()

    return tensor


def raft_infer(img1_path, img2_path, max_size=512):
    with torch.no_grad():
        t1 = load_image_as_tensor(img1_path, max_size=max_size)
        t2 = load_image_as_tensor(img2_path, max_size=max_size)

        padder = InputPadder(t1.shape)
        t1, t2 = padder.pad(t1, t2)

        _, flow = raft_model(t1, t2, iters=10, test_mode=True)
        flow = padder.unpad(flow)

        return flow


def warp(image, flow):
    b, c, h, w = image.shape

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=image.device),
        torch.arange(w, device=image.device),
        indexing="ij"
    )

    grid = torch.stack((grid_x, grid_y), dim=2)
    grid = grid.type_as(image).unsqueeze(0)

    flow = flow.permute(0, 2, 3, 1).type_as(image)
    new_grid = grid + flow

    new_grid[..., 0] = 2 * new_grid[..., 0] / (w - 1) - 1
    new_grid[..., 1] = 2 * new_grid[..., 1] / (h - 1) - 1

    warped = torch.nn.functional.grid_sample(
        image,
        new_grid,
        align_corners=True
    )

    return warped


# -----------------------------
# Main function
# -----------------------------
def run_deoldify_raft(
    input_video_path,
    output_video_path,
    render_factor=21,
    raft_max_size=512,
    blend_current = 0.75,
    blend_previous = 0.25):

    print(f"\n🎬 Processing: {input_video_path}")

    base_name = os.path.splitext(os.path.basename(input_video_path))[0]

    frames_dir = os.path.join("video", "temp_frames", base_name)
    output_dir = os.path.join("video", "temp_output", base_name)

    shutil.rmtree(frames_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # -----------------------------
    # Extract frames
    # -----------------------------
    print("📹 Extracting frames...")

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imwrite(
            os.path.join(frames_dir, f"{frame_idx:05d}.png"),
            frame
        )

        frame_idx += 1

    cap.release()

    frame_files = sorted(os.listdir(frames_dir))
    total_frames = len(frame_files)

    print(f"✅ Extracted {total_frames} frames")

    # -----------------------------
    # Process frames
    # -----------------------------
    prev_colored = None

    for idx in range(total_frames):
        current_frame = os.path.join(frames_dir, frame_files[idx])

        if idx < total_frames - 1:
            next_frame = os.path.join(frames_dir, frame_files[idx + 1])
        else:
            next_frame = current_frame

        print(f"🎨 Frame {idx + 1}/{total_frames}")

        # DeOldify current frame
        color_img = colorizer.get_transformed_image(
            current_frame,
            render_factor=render_factor
        )

        color_np = np.array(color_img)

        color_tensor = torch.from_numpy(color_np)
        color_tensor = color_tensor.permute(2, 0, 1).float()[None]
        color_tensor = color_tensor.to(DEVICE).half()

        # Temporal smoothing with RAFT
        if prev_colored is not None:
            flow = raft_infer(
                current_frame,
                next_frame,
                max_size=raft_max_size
            )

            if flow.shape[-2:] != color_tensor.shape[-2:]:
                flow = torch.nn.functional.interpolate(
                    flow,
                    size=color_tensor.shape[-2:],
                    mode="bilinear",
                    align_corners=True
                )

            warped_prev = warp(prev_colored, flow)

            color_tensor = (
                blend_current * color_tensor +
                blend_previous * warped_prev
            )

        prev_colored = color_tensor

        output_img = color_tensor[0].permute(1, 2, 0).cpu().numpy()
        output_img = np.clip(output_img, 0, 255).astype(np.uint8)

        cv2.imwrite(
            os.path.join(output_dir, f"{idx:05d}.png"),
            cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        )

        torch.cuda.empty_cache()

    # -----------------------------
    # Rebuild video
    # -----------------------------
    print("🎞 Rebuilding video...")

    first_frame = cv2.imread(os.path.join(output_dir, "00000.png"))
    height, width = first_frame.shape[:2]

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 24,
        (width, height)
    )

    for file_name in sorted(os.listdir(output_dir)):
        frame = cv2.imread(os.path.join(output_dir, file_name))
        out.write(frame)

    out.release()

    print(f"✅ Saved: {output_video_path}")

if __name__ == "__main__":
    import sys

    input_video = sys.argv[1]
    output_video = sys.argv[2]

    if len(sys.argv) > 3:
        render_factor = int(sys.argv[3])
    else:
        render_factor = 21

    run_deoldify_raft(
        input_video_path=input_video,
        output_video_path=output_video,
        render_factor=render_factor
    )
