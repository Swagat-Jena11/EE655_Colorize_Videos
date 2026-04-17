# deoldify_noraft.py

import os
import shutil

from deoldify import device
from deoldify.device_id import DeviceId

device.set(device=DeviceId.GPU0)

from deoldify.visualize import get_video_colorizer

colorizer = get_video_colorizer()


def run_deoldify(
    input_video_path,
    output_video_path,
    render_factor=21
):
    print(f"\n🎬 Processing: {input_video_path}")

    file_name = os.path.basename(input_video_path)
    file_name_ext = file_name

    source_dir = "video/source"
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    temp_input_path = os.path.join(source_dir, file_name)

    if not os.path.exists(temp_input_path):
        shutil.copy(input_video_path, temp_input_path)

    print("🎨 Running DeOldify video colorization...")

    result_path = colorizer.colorize_from_file_name(
        file_name_ext,
        render_factor=render_factor
    )

    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    shutil.move(result_path, output_video_path)

    print(f"✅ Saved: {output_video_path}")


if __name__ == "__main__":
    import sys

    input_video = sys.argv[1]
    output_video = sys.argv[2]

    if len(sys.argv) > 3:
        render_factor = int(sys.argv[3])
    else:
        render_factor = 21

    run_deoldify(
        input_video_path=input_video,
        output_video_path=output_video,
        render_factor=render_factor
    )