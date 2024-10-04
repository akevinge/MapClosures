# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import sys
from pathlib import Path
from typing import Sequence, Any

import natsort
import numpy as np


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

    return rot_matrix


class RosbagDataset:
    def __init__(self, data_dir: Sequence[Path], topic: str, gt_topic: str, *_, **__):
        """ROS1 / ROS2 bagfile dataloader.

        It can take either one ROS2 bag file or one or more ROS1 bag files belonging to a split bag.
        The reader will replay ROS1 split bags in correct timestamp order.

        TODO: Merge mcap and rosbag dataloaders into 1
        """
        try:
            from rosbags.highlevel import AnyReader
        except ModuleNotFoundError:
            print('[ERROR] rosbags library not installed, run "pip install -U rosbags"')
            sys.exit(1)

        from kiss_icp.tools.point_cloud2 import read_point_cloud

        self.read_point_cloud = read_point_cloud

        # FIXME: This is quite hacky, trying to guess if we have multiple .bag, one or a dir
        if isinstance(data_dir, Path):
            self.sequence_id = os.path.basename(data_dir).split(".")[0]
            self.bag = AnyReader([data_dir])
        else:
            self.sequence_id = os.path.basename(data_dir[0]).split(".")[0]
            self.bag = AnyReader(data_dir)
            print("[INFO] Reading multiple .bag files in directory:")
            print("\n".join(natsort.natsorted([path.name for path in self.bag.paths])))
        self.bag.open()
        self.topic = self.check_topic(topic)
        self.n_scans = self.bag.topics[self.topic].msgcount

        # limit connections to selected topic
        self.scan_msg_iterator = self.get_scan_msg_iterator()
        self.timestamps = []

        self.gt_topic = self.check_gt_topic(gt_topic)
        self.gt_poses = self.load_gt_poses(self.gt_topic)

        # Visualization Options
        self.use_global_visualizer = True

    def __del__(self):
        if hasattr(self, "bag"):
            self.bag.close()

    def __len__(self):
        return self.n_scans

    def __getitem__(self, idx):
        connection, timestamp, rawdata = next(self.scan_msg_iterator)
        self.timestamps.append(self.to_sec(timestamp))
        msg = self.bag.deserialize(rawdata, connection.msgtype)
        return self.read_point_cloud(msg)

    @staticmethod
    def to_sec(nsec: int):
        return float(nsec) / 1e9

    def get_frames_timestamps(self) -> list:
        return self.timestamps

    def get_scan_msg_iterator(self):
        connections = [x for x in self.bag.connections if x.topic == self.topic]
        return self.bag.messages(connections=connections)

    def load_gt_poses(self, gt_topic: str) -> np.ndarray:
        """Load ground truth poses from a rosbag."""

        # Load ground truth timestamps and poses from bagfile.

        gt_poses: list[Any] = []
        gt_timestamps: list[int] = []
        connections = [x for x in self.bag.connections if x.topic == gt_topic]
        for connection, timestamp, rawdata in self.bag.messages(connections=connections):
            msg = self.bag.deserialize(rawdata, connection.msgtype)
            gt_poses.append(msg.pose.pose)
            gt_timestamps.append(timestamp)

        gt_timestamps = np.asarray(gt_timestamps)

        filtered_gt_poses: list[Any] = []
        if len(gt_poses) > self.n_scans:
            print(
                "[WARNING] The number of GT poses is greater than the number of scans in the bagfile. A GT pose will be selected by nearest timestamp for each scan."
            )

            scan_msg_it = self.get_scan_msg_iterator()
            for _ in range(self.n_scans):
                # Get the timestamp of next scan message.
                _, scan_timestamp, _ = next(scan_msg_it)

                # Find the nearest GT pose to the scan timestamp.
                nearest_idx = np.argmin(np.abs(gt_timestamps - scan_timestamp))

                # Append the nearest GT pose to the filtered GT poses list.
                filtered_gt_poses.append(gt_poses[nearest_idx])

            print("[INFO] GT poses filtered by nearest timestamp.")
            assert len(filtered_gt_poses) == self.n_scans
        else:
            filtered_gt_poses = gt_poses

        # Convert the poses to a 4x4 transformation matrix
        rot = np.array(
            [
                quaternion_rotation_matrix(
                    [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                )
                for pose in filtered_gt_poses
            ]
        )
        trans = np.array(
            [[pose.position.x, pose.position.y, pose.position.z] for pose in filtered_gt_poses]
        )

        T = np.zeros((len(filtered_gt_poses), 4, 4))
        T[:, :3, :3] = rot
        T[:, :3, 3] = trans
        T[:, 3, 3] = 1

        return T

    def check_gt_topic(self, gt_topic: str) -> str:
        # Extract all Odometry msg topics from the bagfile
        odometry_topics = [
            topic[0]
            for topic in self.bag.topics.items()
            if topic[1].msgtype == "nav_msgs/msg/Odometry"
        ]

        def print_available_topics_and_exit():
            print(50 * "-")
            for t in odometry_topics:
                print(f"--gt_topic {t}")
            print(50 * "-")
            sys.exit(1)

        if gt_topic and gt_topic in odometry_topics:
            return gt_topic
        # when user specified the topic check that exists
        if gt_topic and gt_topic not in odometry_topics:
            print(
                f'[ERROR] Dataset does not containg any msg with the topic name "{gt_topic}". '
                "Please select one of the following topics with the --gt_topic flag"
            )
            print_available_topics_and_exit()
        if len(odometry_topics) > 1:
            print(
                "[ERROR] Multiple nav_msgs/msg/Odometry topics available."
                "Please select one of the following topics with the --gt_topic flag"
            )
            print_available_topics_and_exit()

        if len(odometry_topics) == 0:
            print("[ERROR] Your dataset does not contain any nav_msgs/msg/Odometry topic")
        if len(odometry_topics) == 1:
            return odometry_topics[0]

    def check_topic(self, topic: str) -> str:
        # Extract all PointCloud2 msg topics from the bagfile
        point_cloud_topics = [
            topic[0]
            for topic in self.bag.topics.items()
            if topic[1].msgtype == "sensor_msgs/msg/PointCloud2"
        ]

        def print_available_topics_and_exit():
            print(50 * "-")
            for t in point_cloud_topics:
                print(f"--topic {t}")
            print(50 * "-")
            sys.exit(1)

        if topic and topic in point_cloud_topics:
            return topic
        # when user specified the topic check that exists
        if topic and topic not in point_cloud_topics:
            print(
                f'[ERROR] Dataset does not containg any msg with the topic name "{topic}". '
                "Please select one of the following topics with the --topic flag"
            )
            print_available_topics_and_exit()
        if len(point_cloud_topics) > 1:
            print(
                "[ERROR] Multiple sensor_msgs/msg/PointCloud2 topics available."
                "Please select one of the following topics with the --topic flag"
            )
            print_available_topics_and_exit()

        if len(point_cloud_topics) == 0:
            print("[ERROR] Your dataset does not contain any sensor_msgs/msg/PointCloud2 topic")
        if len(point_cloud_topics) == 1:
            return point_cloud_topics[0]
