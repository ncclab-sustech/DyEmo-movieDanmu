import os
import cv2
import subprocess

def split_video(
    video_path,
    output_dir,
    window_sec=10,
    step_sec=10
):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps
    cap.release()

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    num_windows = int(duration // step_sec)

    for i in range(num_windows):
        start = i * step_sec
        output_path = os.path.join(
            output_dir, f"{start}s_to_{start+window_sec}s.mp4"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(start),
            "-t", str(window_sec),
            "-c", "copy",
            output_path
        ]

        print(f"[CUT] {output_path}")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    VIDEO_PATH = "/mnt/dataset1/xinke/Danmu/Code_submit/VLMs/Videos/test.mp4"   # ← 改成你的视频
    OUTPUT_DIR = "Video_10s_clips"

    split_video(VIDEO_PATH, OUTPUT_DIR)