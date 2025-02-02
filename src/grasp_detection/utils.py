from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
import numpy as np
import pandas as pd
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from numpy.typing import NDArray


def to_numpy_array(row: pd.Series, landmark_name: str) -> NDArray:
    return np.array(
        [
            row[f"{landmark_name}_X"],
            row[f"{landmark_name}_Y"],
            row[f"{landmark_name}_Z"],
        ]
    ).reshape((3,))


def set_offset(row) -> NDArray:
    wrist_pos = to_numpy_array(row, "WRIST")
    wrist_as_row = np.tile(wrist_pos, [21, 1]).reshape(-1)
    return row - wrist_as_row


def normalize(row) -> NDArray:
    norm = np.linalg.norm(
        to_numpy_array(row, "PINKY_MCP") - to_numpy_array(row, "WRIST")
    )

    return row / norm


def align(row: pd.Series):

    landmark_names = [
        "WRIST",
        "THUMB_CMC",
        "THUMB_MCP",
        "THUMB_IP",
        "THUMB_TIP",
        "INDEX_FINGER_MCP",
        "INDEX_FINGER_PIP",
        "INDEX_FINGER_DIP",
        "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP",
        "MIDDLE_FINGER_PIP",
        "MIDDLE_FINGER_DIP",
        "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP",
        "RING_FINGER_PIP",
        "RING_FINGER_DIP",
        "RING_FINGER_TIP",
        "PINKY_MCP",
        "PINKY_PIP",
        "PINKY_DIP",
        "PINKY_TIP",
    ]

    wrist_pinky_v = to_numpy_array(row, "PINKY_MCP") - to_numpy_array(
        row, "WRIST"
    )
    wrist_index_v = to_numpy_array(row, "INDEX_FINGER_MCP") - to_numpy_array(
        row, "WRIST"
    )

    palm_meridian_v = (
        to_numpy_array(row, "PINKY_MCP")
        + (
            to_numpy_array(row, "INDEX_FINGER_MCP")
            - to_numpy_array(row, "PINKY_MCP")
        )
        / 2
    )

    try:
        palm_normal = np.cross(wrist_pinky_v, wrist_index_v)
    except ValueError as e:
        print(e)

    T_mat = np.eye(4)
    try:
        T_mat[:3, :3] = np.c_[
            palm_normal,
            np.cross(palm_normal, palm_meridian_v),
            palm_meridian_v
            # palm_normal, np.cross(palm_normal, wrist_pinky_v), wrist_pinky_v
        ]
    except ValueError as e:
        print(e)
    yaw_rot = np.array(
        [
            [np.cos(-np.pi / 2), -np.sin(-np.pi / 2), 0, 0],
            [np.sin(-np.pi / 2), np.cos(-np.pi / 2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    dict_row = {}
    for name in landmark_names:
        # Adding 1 for homogeneous transform
        landmark_value = np.r_[to_numpy_array(row, name), [1]]
        transformed_value = np.matmul(landmark_value, T_mat)
        transformed_value = np.matmul(yaw_rot, transformed_value)
        dict_row[f"{name}_X"] = transformed_value[0]
        dict_row[f"{name}_Y"] = transformed_value[1]
        dict_row[f"{name}_Z"] = transformed_value[2]

    return pd.Series(dict_row)


def rotate_180(row):

    landmark_names = [
        "WRIST",
        "THUMB_CMC",
        "THUMB_MCP",
        "THUMB_IP",
        "THUMB_TIP",
        "INDEX_FINGER_MCP",
        "INDEX_FINGER_PIP",
        "INDEX_FINGER_DIP",
        "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP",
        "MIDDLE_FINGER_PIP",
        "MIDDLE_FINGER_DIP",
        "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP",
        "RING_FINGER_PIP",
        "RING_FINGER_DIP",
        "RING_FINGER_TIP",
        "PINKY_MCP",
        "PINKY_PIP",
        "PINKY_DIP",
        "PINKY_TIP",
    ]

    T_mat = np.array(
        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    dict_row = {}
    for name in landmark_names:
        # Adding 1 for homogeneous transform
        landmark_value = np.r_[to_numpy_array(row, name), [1]]
        transformed_value = np.matmul(landmark_value, T_mat)
        dict_row[f"{name}_X"] = transformed_value[0]
        dict_row[f"{name}_Y"] = transformed_value[1]
        dict_row[f"{name}_Z"] = transformed_value[2]

    return pd.Series(dict_row)


def np_to_Point(np_array):
    p = Point()
    p.x = np_array[0]
    p.y = np_array[1]
    p.z = np_array[2]
    return p


def draw_vector(
    m_pb, id, start, end, parent_frame, namespace="debug", color=(1, 0, 0)
):
    marker = Marker()

    marker.header.frame_id = parent_frame
    marker.header.stamp = rospy.Time.now()
    marker.ns = namespace
    marker.id = id

    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    marker.points.append(np_to_Point(start))
    marker.points.append(np_to_Point(end))

    marker.scale.x = 0.01
    marker.scale.y = 0.02
    marker.scale.z = 0

    marker.pose.orientation.w = 1.0

    marker.color.a = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]

    m_pb.publish(marker)


def pub_landmark(br, name, pos, parent_name):
    ts_msg = TransformStamped()
    ts_msg.header.stamp = rospy.Time.now()
    ts_msg.header.frame_id = parent_name
    ts_msg.child_frame_id = name

    ts_msg.transform.translation.x = pos[0]
    ts_msg.transform.translation.y = pos[1]
    ts_msg.transform.translation.z = pos[2]

    ts_msg.transform.rotation.x = 0
    ts_msg.transform.rotation.y = 0
    ts_msg.transform.rotation.z = 0
    ts_msg.transform.rotation.w = 1

    br.sendTransform(ts_msg)


def landmark_pos_from_sample(sample, name):
    if name == "WRIST":
        return np.zeros(3)
    return np.array(
        [sample[f"{name}_X"], sample[f"{name}_Y"], sample[f"{name}_Z"]]
    )


def pub_sample(br, sample, parents, handedness):
    published_landmarks = []
    for column_name, value in sample.iteritems():
        name = column_name[:-2]
        if name in published_landmarks:
            continue

        if name == "WRIST":
            pos_in_parent = np.zeros(3)
            parent_name = handedness + parents[name]
        else:
            parent_name = parents[name]
            lmp_p_array = landmark_pos_from_sample(sample, parent_name)
            lm_p_array = landmark_pos_from_sample(sample, name)
            pos_in_parent = lm_p_array - lmp_p_array
        pub_landmark(
            br,
            name.lower(),
            pos_in_parent,
            parent_name.lower(),
        )
        published_landmarks.append(name)


def unit_vector(vector: NDArray) -> NDArray:
    """Returns the unit vector of the vector.

    Args:
        vector (np.array): The vector

    Returns:
        np.array: The unit vector
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1: np.array, v2: np.array) -> float:
    """Returns the angle in radians between vectors 'v1' and 'v2'

    Examples:
    - angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    - angle_between((1, 0, 0), (1, 0, 0))
    0.0
    -angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793

    Args:
        v1 (np.array): Vector 1
        v2 (np.array): Vector 2

    Returns:
        np.array: The unit vector
    """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def pitch_mat(pitch_angle):
    return np.array(
        [
            [np.cos(pitch_angle), 0, np.sin(pitch_angle)],
            [0, 1, 0],
            [-np.sin(pitch_angle), 0, np.cos(pitch_angle)],
        ]
    )


def roll_mat(roll_angle):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll_angle), -np.sin(roll_angle)],
            [0, np.sin(roll_angle), np.cos(roll_angle)],
        ]
    )


def yaw_mat(yaw_angle):
    return np.array(
        [
            [np.cos(yaw_angle), -np.sin(yaw_angle), 0],
            [np.sin(yaw_angle), np.cos(yaw_angle), 0],
            [0, 0, 1],
        ]
    )


def get_finger_joint(
    v_wrist_0: np.array,
    v_0_1: np.array,
    v_1_2: np.array,
    v_2_3: np.array,
    v_0_3: np.array,
    is_thumb: bool,
    is_left: bool = True,
    m_pub=None,
):
    joints = {"meta": {"abdu": 0, "flex": 0}, "proximal": {"flex": 0}}
    if is_thumb:
        joints["distal"] = {"flex": 0}
    else:
        joints["middle"] = {"flex": 0}

    # if m_pub is not None:
    #     # marker_anchor = v_wrist_0
    #     marker_anchor = np.array([-1, 0, 0])
    if is_thumb:
        v_0_1 = v_wrist_0 + v_0_1
        L1 = np.linalg.norm(v_0_1)
        if is_left:
            meta_flex = np.arcsin(v_0_1[2] / L1)  # theta 2
            meta_abdu = np.arcsin(
                -v_0_1[1] / (L1 * np.cos(meta_flex))
            )  # theta 1
        else:
            # Swap abdu (old piv) and flex
            meta_abdu = np.arcsin(-v_0_1[1] / L1)  # theta 2
            meta_flex = np.arcsin(
                v_0_1[2] / (L1 * np.cos(meta_abdu))
            )  # theta 1

    else:
        L1 = np.linalg.norm(v_0_1)
        meta_abdu = np.arcsin(v_0_1[0] / L1)  # theta 2
        meta_flex = np.arcsin(-v_0_1[1] / (L1 * np.cos(meta_abdu)))
        meta_abdu *= -1

    # meta_abdu = 0
    handedness_fix = 1.0
    if not is_thumb:
        handedness_fix = -1.0
    # joints["meta"]["flex"] = angle_between(v_0_1, v_wrist_0)
    joints["meta"]["flex"] = meta_flex
    joints["meta"]["abdu"] = handedness_fix * meta_abdu
    joints["proximal"]["flex"] = angle_between(v_1_2, v_0_1)
    if is_thumb:
        last_joint = "distal"
    else:
        last_joint = "middle"
    joints[last_joint]["flex"] = angle_between(v_2_3, v_1_2)
    return joints


def get_finger_vectors(sample, finger_name):
    landmark_finger_name = finger_name.upper()
    if finger_name != "pinky" and finger_name != "thumb":
        landmark_finger_name += "_FINGER"

    # landmark 0 = first finger landmark. landmark 3 = last finger landmark
    if finger_name == "thumb":
        landmark_0 = "CMC"
        landmark_1 = "MCP"
        landmark_2 = "IP"
    else:
        landmark_0 = "MCP"
        landmark_1 = "PIP"
        landmark_2 = "DIP"

    v_2_3 = to_numpy_array(
        sample, f"{landmark_finger_name}_TIP"
    ) - to_numpy_array(sample, f"{landmark_finger_name}_{landmark_2}")

    v_1_2 = to_numpy_array(
        sample, f"{landmark_finger_name}_{landmark_2}"
    ) - to_numpy_array(sample, f"{landmark_finger_name}_{landmark_1}")

    v_0_1 = to_numpy_array(
        sample, f"{landmark_finger_name}_{landmark_1}"
    ) - to_numpy_array(sample, f"{landmark_finger_name}_{landmark_0}")

    v_0_3 = to_numpy_array(
        sample, f"{landmark_finger_name}_TIP"
    ) - to_numpy_array(sample, f"{landmark_finger_name}_{landmark_0}")

    v_wrist_0 = to_numpy_array(
        sample, f"{landmark_finger_name}_{landmark_0}"
    ) - to_numpy_array(sample, "WRIST")

    return v_wrist_0, v_0_1, v_1_2, v_2_3, v_0_3


def get_hand_joint_angles(sample, is_left, prefix="", m_pub=None):
    joints = {}

    if prefix != "":
        prefix = prefix + "/"

    joint_types = ["abdu", "flex"]
    link_types = {
        "thumb": ["meta", "proximal", "distal"],
        "non-thumb": ["meta", "proximal", "middle"],
    }
    fingers_types = ["thumb", "index", "middle", "ring", "pinky"]

    for finger in fingers_types:
        if finger != "thumb":
            finger_type = "non-thumb"
        else:
            finger_type = "thumb"
        if finger == "ring":
            finger_joints = get_finger_joint(
                *get_finger_vectors(sample, finger),
                finger == "thumb",
                is_left=is_left,
                m_pub=m_pub,
            )
        else:
            finger_joints = get_finger_joint(
                *get_finger_vectors(sample, finger),
                finger == "thumb",
                is_left=is_left,
                m_pub=None,
            )
        for link in link_types[finger_type]:
            for joint in joint_types:
                if link != "meta" and joint == "abdu":
                    continue
                joints[f"{prefix}{finger}_{link}_{joint}"] = finger_joints[
                    link
                ][joint]
                # if joint == "abdu" or finger == "thumb":
                #     joints[f"{prefix}{finger}_{link}_{joint}"] = 0

    return joints


def publish_joint_states(js_pub, joint_angles):
    js = JointState()
    for name, value in joint_angles.items():
        js.name.append(name)
        js.position.append(value)
    js.header.stamp = rospy.Time.now()
    js_pub.publish(js)
