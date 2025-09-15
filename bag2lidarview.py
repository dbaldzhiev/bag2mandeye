#!/usr/bin/env python3
"""Convert a ROS2 bag (sqlite) to MCAP and export IMU data for LidarView.

The script writes all messages from the input bag into a single MCAP file and
creates a CSV containing IMU readings with additional pose information. The
CSV columns follow the LidarView convention:

``time, odom, acc_x, acc_y, acc_z, w_x, w_y, w_z, x, y, z, roll, pitch, yaw``

`odom` is the distance travelled based on the odometry pose. Orientation is
expressed as Euler angles in YXZ order (roll, pitch, yaw).

Example:

```
python bag2lidarview.py mybag out.mcap imu.csv \
    --imu_topic /imu/data --odom_topic /wheel/odometry
```
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass

from mcap_ros2.writer import Writer
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation


TYPESTORE = get_typestore(Stores.ROS2_FOXY)


def get_sec(time) -> float:
    """Return ROS time in seconds."""

    return float(time.sec) + float(time.nanosec) / 1e9


@dataclass
class PoseState:
    """Container for pose values used in the output CSV."""

    odom: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


def convert_bag(
    bag_path: str,
    output_mcap: str,
    output_csv: str,
    imu_topic: str,
    odom_topic: str,
) -> None:
    """Convert rosbag2 to MCAP and export IMU data."""

    with Reader(bag_path) as reader, open(output_mcap, "wb") as mcap_file, open(
        output_csv, "w", newline=""
    ) as csv_file:
        writer = Writer(mcap_file)

        # prepare MCAP channels
        channel_ids: dict[int, int] = {}
        for conn in reader.connections:
            schema_id = writer.register_schema(
                name=conn.msgtype, encoding="ros2msg", data=conn.msgdef.encode()
            )
            channel_ids[conn.id] = writer.register_channel(
                schema_id=schema_id, topic=conn.topic, message_encoding="cdr"
            )

        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "time",
                "odom",
                "acc_x",
                "acc_y",
                "acc_z",
                "w_x",
                "w_y",
                "w_z",
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
            ]
        )

        pose = PoseState()

        for conn, ts, raw in reader.messages():
            # write to MCAP
            writer.write_message(
                channel_id=channel_ids[conn.id],
                log_time=ts,
                publish_time=ts,
                data=raw,
            )

            if conn.topic == odom_topic:
                msg = TYPESTORE.deserialize_cdr(raw, conn.msgtype)
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                z = msg.pose.pose.position.z
                q = msg.pose.pose.orientation
                roll, pitch, yaw = Rotation.from_quat(
                    [q.x, q.y, q.z, q.w]
                ).as_euler("yxz", degrees=False)
                pose = PoseState(
                    odom=math.sqrt(x * x + y * y + z * z),
                    x=x,
                    y=y,
                    z=z,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                )

            elif conn.topic == imu_topic:
                msg = TYPESTORE.deserialize_cdr(raw, conn.msgtype)
                csv_writer.writerow(
                    [
                        get_sec(msg.header.stamp),
                        pose.odom,
                        msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z,
                        msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z,
                        pose.x,
                        pose.y,
                        pose.z,
                        pose.roll,
                        pose.pitch,
                        pose.yaw,
                    ]
                )

        writer.finish()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a ROS2 bag to MCAP and export IMU data for LidarView"
    )
    parser.add_argument("input_bag", help="Path to ROS2 bag (directory)")
    parser.add_argument("output_mcap", help="Output MCAP file path")
    parser.add_argument("output_csv", help="Output CSV file path")
    parser.add_argument("--imu_topic", default="/imu/data")
    parser.add_argument("--odom_topic", default="/odom")
    args = parser.parse_args()

    convert_bag(
        args.input_bag,
        args.output_mcap,
        args.output_csv,
        args.imu_topic,
        args.odom_topic,
    )


if __name__ == "__main__":
    main()

