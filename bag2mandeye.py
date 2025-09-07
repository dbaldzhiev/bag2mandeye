#!/usr/bin/env python3
"""Convert a ROS2 bag to Mandeye dataset.

This script reads a ROS2 bag (sqlite-based) and writes Mandeye-style
files matching Mandeye controller layout. For each chunk the script writes
``lidarXXXX.laz`` (point cloud), ``imuXXXX.csv`` (IMU) and
``lidarXXXX.sn`` containing lidar serial numbers. The format was inferred
from https://github.com/MapsHD/mandeye_to_bag and
https://github.com/JanuszBedkowski/mandeye_controller.

Dependencies:
  * rosbags - for reading rosbag2 files without ROS2 runtime
  * laspy   - for writing LAZ point cloud files
  * yaml    - for reading bag metadata (PyYAML)
  * tqdm    - optional, displays progress bar

For LAZ compression support install laspy with a backend, e.g.:

    pip install laspy[lazrs]

Example usage:
  python bag2mandeye.py mybag /tmp/out \
      --pointcloud_topic /livox/lidar --imu_topic /imu/data \
      --lidar_sn ABC123
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import laspy
import yaml
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback if tqdm not installed
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


FLOAT32 = 7  # sensor_msgs/msg/PointField datatype for float32
TYPESTORE = get_typestore(Stores.ROS2_FOXY)


def get_sec(time) -> float:
    """Return ROS time in seconds."""
    return float(time.sec) + float(time.nanosec) / 1e9


def get_nano(time) -> int:
    """Return ROS time in nanoseconds."""
    return int(time.sec) * 1_000_000_000 + int(time.nanosec)


def get_interpolated_ts(frame_rate: float, start_ts: float, num_points: int, point_number: int) -> float:
    """Interpolate timestamp for lidar point.

    Args:
        frame_rate: Average frame time of lidar (seconds).
        start_ts:   Timestamp of first point (seconds).
        num_points: Number of points in scan.
        point_number: Index of point.
    Returns:
        Timestamp in seconds.
    """
    return start_ts + (point_number * frame_rate) / num_points


@dataclass
class MandeyePoint:
    """Simple container for a Mandeye point."""
    x: float
    y: float
    z: float
    intensity: float
    timestamp: int  # nanoseconds


def parse_pointcloud(msg) -> np.ndarray:
    """Extract x, y, z, intensity from a PointCloud2 message.

    Returns a structured numpy array with fields x, y, z, intensity.
    """
    required = {"x", "y", "z", "intensity"}
    fieldmap = {f.name: f for f in msg.fields if f.datatype == FLOAT32}
    if not required.issubset(fieldmap):
        missing = required - set(fieldmap)
        raise ValueError(f"PointCloud2 missing fields: {missing}")

    dtype = np.dtype({
        "names": ["x", "y", "z", "intensity"],
        "formats": ["<f4"] * 4,
        "offsets": [fieldmap[n].offset for n in ["x", "y", "z", "intensity"]],
        "itemsize": msg.point_step,
    })
    return np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)


def save_data(
    output_directory: str,
    count: int,
    points: Sequence[MandeyePoint],
    imu_lines: Sequence[str],
    lidar_sn: str,
    lidar_id: int,
) -> None:
    """Write buffered data to disk in Mandeye format."""
    os.makedirs(output_directory, exist_ok=True)

    lidar_path = os.path.join(output_directory, f"lidar{count:04d}.laz")
    imu_path = os.path.join(output_directory, f"imu{count:04d}.csv")
    sn_path = os.path.join(output_directory, f"lidar{count:04d}.sn")

    written_lidar_path = None
    if points:
        xs = np.array([p.x for p in points], dtype=np.float64)
        ys = np.array([p.y for p in points], dtype=np.float64)
        zs = np.array([p.z for p in points], dtype=np.float64)
        intensities = np.array([p.intensity for p in points], dtype=np.float32)
        times = np.array([p.timestamp for p in points], dtype=np.float64) / 1e9

        header = laspy.LasHeader(point_format=1, version="1.2")
        header.scales = [0.0001, 0.0001, 0.0001]
        header.offsets = [float(xs.min()), float(ys.min()), float(zs.min())]

        las = laspy.LasData(header)
        las.x = xs
        las.y = ys
        las.z = zs
        las.intensity = intensities
        las.gps_time = times
        available = laspy.LazBackend.detect_available()
        start = time.time()
        if available:
            las.write(lidar_path, laz_backend=available[0])
            written_lidar_path = lidar_path
        else:
            las_path = os.path.splitext(lidar_path)[0] + ".las"
            las.write(las_path)
            written_lidar_path = las_path
            print(
                "No LAZ backend detected; wrote uncompressed LAS file",
                os.path.basename(las_path),
            )
        duration = time.time() - start

        status = {
            "arch": "",
            "fs_benchmark": {"write_speed_10mb": 0.0, "write_speed_1mb": 0.0},
            "gnss": None,
            "hardware": "",
            "hash": "",
            "lastLazStatus": {
                "decimation_step": 1,
                "filename": os.path.abspath(written_lidar_path),
                "points_count": len(points),
                "save_duration_sec1": duration,
                "save_duration_sec2": duration,
                "size_mb": os.path.getsize(written_lidar_path) / (1024 * 1024),
            },
            "livox": None,
            "name": "Mandeye",
            "state": "SCANNING",
            "version": "",
        }
        status_path = os.path.join(output_directory, "status.json")
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4)

    with open(imu_path, "w", encoding="utf-8") as f:
        header = "timestamp gyroX gyroY gyroZ accX accY accZ imuId timestampUnix\n"
        f.write(header)
        if imu_lines:
            f.write("\n".join(imu_lines) + "\n")

    with open(sn_path, "w", encoding="utf-8") as f:
        f.write(f"{lidar_id} {lidar_sn}\n")


def get_message_count(bag_path: str) -> int:
    """Return total number of messages in the bag or 0 if unknown."""
    metadata_path = os.path.join(bag_path, "metadata.yaml")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return int(data["rosbag2_bagfile_information"]["message_count"])
    except Exception:
        return 0


def compute_lidar_frame_rate(bag_path: str, topic: str) -> float:
    """Estimate lidar frame rate in seconds from pointcloud header timestamps."""
    header_diffs: List[float] = []
    last_header_ts = 0.0
    start_ts = 0.0
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        for conn, _, raw in reader.messages(connections=connections):
            msg = TYPESTORE.deserialize_cdr(raw, conn.msgtype)
            ts = get_sec(msg.header.stamp)
            if last_header_ts != 0.0:
                diff = ts - last_header_ts
                if diff > 0.0:
                    header_diffs.append(diff)
                if ts - start_ts > 20.0:
                    break
            else:
                start_ts = ts
            last_header_ts = ts
    if not header_diffs:
        return 0.0
    return sum(header_diffs) / len(header_diffs)


def convert_bag_to_mandeye(
    bag_path: str,
    output_directory: str,
    pointcloud_topic: str,
    imu_topic: str,
    chunk_len: float,
    emulate_point_ts: bool,
    lidar_sn: str,
    lidar_id: int,
) -> None:
    """Main conversion routine."""
    lidar_frame_rate = compute_lidar_frame_rate(bag_path, pointcloud_topic) if emulate_point_ts else 0.0

    buffer_points: List[MandeyePoint] = []
    buffer_imu: List[str] = []
    last_save_timestamp = 0.0
    last_imu_timestamp = -1.0
    count = 0

    total_messages = get_message_count(bag_path)

    with Reader(bag_path) as reader:
        iterator = reader.messages()
        iterator = tqdm(
            iterator,
            total=total_messages or None,
            desc="Converting",
        )
        for connection, ts, raw in iterator:
            topic = connection.topic
            if topic == imu_topic:
                msg = TYPESTORE.deserialize_cdr(raw, connection.msgtype)
                nano = get_nano(msg.header.stamp)
                line = (
                    f"{nano} "
                    f"{msg.angular_velocity.x} {msg.angular_velocity.y} {msg.angular_velocity.z} "
                    f"{msg.linear_acceleration.x} {msg.linear_acceleration.y} {msg.linear_acceleration.z} "
                    f"{lidar_id} {nano}"
                )
                buffer_imu.append(line)
                last_imu_timestamp = get_sec(msg.header.stamp)
                if last_save_timestamp == 0.0:
                    last_save_timestamp = last_imu_timestamp
            elif topic == pointcloud_topic and last_imu_timestamp > 0.0:
                msg = TYPESTORE.deserialize_cdr(raw, connection.msgtype)
                ts_sec = get_sec(msg.header.stamp)
                if abs(ts_sec - last_imu_timestamp) < 0.05 * chunk_len:
                    arr = parse_pointcloud(msg)
                    num_points = len(arr)
                    header_ts_sec = ts_sec
                    stamp_base = get_nano(msg.header.stamp)
                    for idx in range(num_points):
                        if emulate_point_ts:
                            pt_ts = int(get_interpolated_ts(lidar_frame_rate, header_ts_sec, num_points, idx) * 1e9)
                        else:
                            pt_ts = stamp_base
                        p = MandeyePoint(
                            float(arr['x'][idx]),
                            float(arr['y'][idx]),
                            float(arr['z'][idx]),
                            float(arr['intensity'][idx]),
                            pt_ts,
                        )
                        buffer_points.append(p)
                else:
                    # skip pointcloud if time difference to imu is too large
                    pass

            message_time_sec = ts / 1e9
            if message_time_sec - last_save_timestamp > chunk_len and last_save_timestamp > 0.0:
                save_data(output_directory, count, buffer_points, buffer_imu, lidar_sn, lidar_id)
                buffer_points.clear()
                buffer_imu.clear()
                last_save_timestamp = message_time_sec
                count += 1

    if buffer_points:
        save_data(output_directory, count, buffer_points, buffer_imu, lidar_sn, lidar_id)


def main():
    parser = argparse.ArgumentParser(description="Convert a ROS2 bag to Mandeye files")
    parser.add_argument("input_bag", help="Path to ROS2 bag (directory with metadata.yaml)")
    parser.add_argument("output_directory", help="Output directory for Mandeye files")
    parser.add_argument("--pointcloud_topic", default="/livox/lidar")
    parser.add_argument("--imu_topic", default="/livox/imu")
    parser.add_argument("--chunk_len", type=float, default=20.0, help="Chunk length in seconds")
    parser.add_argument("--emulate_point_ts", action="store_true", help="Interpolate timestamps for points")
    parser.add_argument("--lidar_sn", default="ABC123", help="Lidar serial number for .sn file")
    parser.add_argument("--lidar_id", type=int, default=0, help="Lidar id for .sn file")
    args = parser.parse_args()

    convert_bag_to_mandeye(
        args.input_bag,
        args.output_directory,
        args.pointcloud_topic,
        args.imu_topic,
        args.chunk_len,
        args.emulate_point_ts,
        args.lidar_sn,
        args.lidar_id,
    )


if __name__ == "__main__":
    main()