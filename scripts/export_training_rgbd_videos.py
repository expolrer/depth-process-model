#!/usr/bin/env python3
"""Export aligned RoboTwin RGB-D training clips for the project site."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import h5py
import numpy as np


SCENES = ("stack_blocks_two", "hanging_mug")
CAMERAS = ("head_camera", "left_camera", "right_camera")
METHODS = ("d0", "d1", "d3")
METHOD_LABELS = {"d0": "D0 CLEAN GT", "d1": "D1 SENSOR NOISE", "d3": "D3 LINGBOT FILLED"}
WIDTH, HEIGHT = 320, 240


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths(root: Path, scene: str, episode: int) -> dict[str, Path]:
    subdir = f"{scene}/depth_master_clean/data"
    return {
        "d0": root / "master" / subdir / f"episode{episode}.hdf5",
        "d1": root / "derived/realsense_d1" / subdir / f"episode{episode}.d1_realsense.hdf5",
        "d3": root / "derived/lingbot_d3" / subdir / f"episode{episode}.d3_lingbot_sensor_fused.hdf5",
    }


def depth_dataset(handle: h5py.File, method: str, camera: str) -> h5py.Dataset:
    key = f"observation/{camera}/depth" if method == "d0" else f"depth/{camera}"
    return handle[key]


def depth_ranges(master: h5py.File, frame_count: int) -> dict[str, tuple[float, float]]:
    ranges = {}
    for camera in CAMERAS:
        dataset = depth_dataset(master, "d0", camera)
        samples = []
        for index in range(0, frame_count, max(1, frame_count // 24)):
            frame = dataset[index, ::4, ::4]
            valid = frame[np.isfinite(frame) & (frame > 0)]
            samples.append(valid.astype(np.float32))
        values = np.concatenate(samples)
        low, high = np.percentile(values, (2, 98))
        margin = max((high - low) * 0.08, 5.0)
        ranges[camera] = (max(0.0, float(low - margin)), float(high + margin))
    return ranges


def colorize(depth_mm: np.ndarray, depth_range: tuple[float, float]) -> tuple[np.ndarray, int]:
    low, high = depth_range
    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    normalized = np.clip((depth_mm.astype(np.float32) - low) / (high - low), 0, 1)
    near_is_warm = np.uint8(np.rint((1 - normalized) * 255))
    image = cv2.applyColorMap(near_is_warm, cv2.COLORMAP_TURBO)
    image[~valid] = (15, 18, 23)
    return image, int(valid.sum())


def decode_rgb(master: h5py.File, camera: str, index: int) -> np.ndarray:
    encoded = bytes(master[f"observation/{camera}/rgb"][index]).rstrip(b"\x00")
    frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape[:2] != (HEIGHT, WIDTH):
        raise ValueError(f"Invalid RGB frame: {camera} {index}")
    return frame


def render_frame(
    master: h5py.File,
    depth_source: h5py.File,
    method: str,
    episode: int,
    index: int,
    frame_count: int,
    ranges: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, dict[str, int]]:
    canvas = np.full((576, WIDTH * 3, 3), (19, 23, 29), dtype=np.uint8)
    valid_counts = {}
    for column, camera in enumerate(CAMERAS):
        left = column * WIDTH
        canvas[40:280, left:left + WIDTH] = decode_rgb(master, camera, index)
        depth, valid_counts[camera] = colorize(depth_dataset(depth_source, method, camera)[index], ranges[camera])
        canvas[312:552, left:left + WIDTH] = depth
        name = ("HEAD", "LEFT WRIST", "RIGHT WRIST")[column]
        cv2.putText(canvas, name + " / RGB", (left + 10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (235, 242, 246), 1, cv2.LINE_AA)
        low, high = ranges[camera]
        scale = f"{low / 1000:.2f}-{high / 1000:.2f} m"
        cv2.putText(canvas, scale, (left + 10, 303), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (168, 183, 196), 1, cv2.LINE_AA)
        if column:
            cv2.line(canvas, (left, 40), (left, 552), (53, 63, 73), 1)
    cv2.putText(canvas, METHOD_LABELS[method], (10, 571), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (206, 229, 241), 1, cv2.LINE_AA)
    progress = f"EPISODE {episode}  /  FRAME {index + 1:03d}/{frame_count:03d}  /  30 FPS"
    cv2.putText(canvas, progress, (620, 571), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (146, 168, 183), 1, cv2.LINE_AA)
    return canvas, valid_counts


def export_scene(root: Path, output: Path, scene: str, episode: int) -> list[dict]:
    paths = source_paths(root, scene, episode)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    scene_output = output / scene
    scene_output.mkdir(parents=True, exist_ok=True)
    records = []
    with h5py.File(paths["d0"], "r") as master, h5py.File(paths["d1"], "r") as d1, h5py.File(paths["d3"], "r") as d3:
        sources = {"d0": master, "d1": d1, "d3": d3}
        count = len(depth_dataset(master, "d0", CAMERAS[0]))
        if not np.array_equal(d1["frame_index"][:], np.arange(count)):
            raise ValueError(f"D1 frame alignment mismatch: {scene}")
        for method in METHODS:
            if not bool(sources[method].attrs.get("complete", True)):
                raise ValueError(f"Incomplete {method} dataset: {scene}")
            for camera in CAMERAS:
                if len(depth_dataset(sources[method], method, camera)) != count:
                    raise ValueError(f"Depth frame count mismatch: {scene} {method} {camera}")
        ranges = depth_ranges(master, count)
        for method in METHODS:
            target = scene_output / f"episode{episode}_{method}.mp4"
            poster = scene_output / f"episode{episode}_{method}.jpg"
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "960x576",
                "-framerate", "30", "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "27", "-threads", "2",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
            ]
            valid_counts = {camera: 0 for camera in CAMERAS}
            with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
                assert process.stdin is not None
                for index in range(count):
                    frame, counts = render_frame(master, sources[method], method, episode, index, count, ranges)
                    process.stdin.write(frame.tobytes())
                    if index == count // 3:
                        if not cv2.imwrite(str(poster), frame, [cv2.IMWRITE_JPEG_QUALITY, 86]):
                            raise RuntimeError(f"Failed to write poster: {poster}")
                    for camera, valid in counts.items():
                        valid_counts[camera] += valid
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError(f"ffmpeg failed: {target}")
            record = {
                "scene": scene,
                "method": method,
                "episode": episode,
                "frames": count,
                "fps": 30,
                "video": target.relative_to(output).as_posix(),
                "poster": poster.relative_to(output).as_posix(),
                "video_bytes": target.stat().st_size,
                "video_sha256": file_sha256(target),
                "depth_valid_fraction": {
                    camera: round(valid_counts[camera] / (count * WIDTH * HEIGHT), 5)
                    for camera in CAMERAS
                },
                "depth_scale_m": {
                    camera: [round(value / 1000, 4) for value in ranges[camera]]
                    for camera in CAMERAS
                },
                "source_files": {key: str(path) for key, path in paths.items()},
            }
            records.append(record)
            print(json.dumps({"scene": scene, "method": method, "bytes": record["video_bytes"]}), flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/ssd/hhw/depth-model/datasets"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for scene in args.scenes:
        records.extend(export_scene(args.data_root, args.output, scene, args.episode))
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
