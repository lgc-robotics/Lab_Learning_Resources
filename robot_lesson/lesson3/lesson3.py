"""Lesson 3：旋转矩阵、RPY 角与机械臂末端位姿可视化。"""

import numpy as np
import matplotlib.pyplot as plt


class RobotKinematics:
    """RML63 六轴机械臂运动学与位姿转换。"""

    def __init__(self, tool_length=140.0):
        self.tool_length = tool_length

    @staticmethod
    def _DHTrans(alpha, a, d, theta):
        """根据一组改进 D-H 参数计算齐次变换矩阵。"""
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, a],
            [
                np.sin(theta) * np.cos(alpha),
                np.cos(theta) * np.cos(alpha),
                -np.sin(alpha),
                -np.sin(alpha) * d,
            ],
            [
                np.sin(theta) * np.sin(alpha),
                np.cos(theta) * np.sin(alpha),
                np.cos(alpha),
                np.cos(alpha) * d,
            ],
            [0, 0, 0, 1],
        ])

    def fkine(self, theta):
        """输入六个关节角（弧度），返回各节矩阵和末端矩阵。"""
        initial_offset = np.array([
            0,
            -np.pi / 2,
            np.pi / 2,
            0,
            np.pi,
            np.pi,
        ])
        th = np.array(theta) + initial_offset
        d = np.array([162.5, 0, 0, 405, 0, 132.3 + self.tool_length])
        a = np.array([0, -86, 380, 69, 0, 0])
        alpha = np.array([
            0,
            -np.pi / 2,
            0,
            np.pi / 2,
            -np.pi / 2,
            -np.pi / 2,
        ])

        matrices = []
        t_final = np.eye(4)

        for index in range(6):
            current_matrix = self._DHTrans(
                alpha[index],
                a[index],
                d[index],
                th[index],
            )
            matrices.append(current_matrix)
            t_final = np.matmul(t_final, current_matrix)

        return matrices, t_final

    @staticmethod
    def _rpy2rotation(rpy):
        """把 RPY 角转换为旋转矩阵，输入单位为弧度。"""
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]

        rotation_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ])
        rotation_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ])
        rotation_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ])

        # RPY 采用 ZYX 组合：R = Rz(yaw) * Ry(pitch) * Rx(roll)。
        rotation = rotation_z @ rotation_y @ rotation_x

        return rotation

    @staticmethod
    def _rotation2rpy(rotation):
        """把旋转矩阵转换为 RPY 角，返回单位为弧度。"""
        if rotation.shape == (4, 4):
            rotation = rotation[0:3, 0:3]

        pitch = np.arctan2(
            -rotation[2, 0],
            np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2),
        )

        if abs(np.cos(pitch)) < 1e-10:
            yaw = 0.0
            roll = np.arctan2(
                np.sign(pitch) * rotation[0, 1],
                rotation[1, 1],
            )
        else:
            roll = np.arctan2(rotation[2, 1], rotation[2, 2])
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])

        return np.array([roll, pitch, yaw])

    def homography2pose(self, homography):
        """把齐次矩阵转换为 [x, y, z, roll, pitch, yaw]。"""
        rpy = self._rotation2rpy(homography)

        pose = np.array([
            homography[0, 3],
            homography[1, 3],
            homography[2, 3],
            rpy[0],
            rpy[1],
            rpy[2],
        ])

        return pose

    def pose2homography(self, pose):
        """把 [x, y, z, roll, pitch, yaw] 转换为齐次矩阵。"""
        homography = np.eye(4)
        homography[0:3, 0:3] = self._rpy2rotation(pose[3:6])
        homography[0:3, 3] = pose[0:3]

        return homography

    @staticmethod
    def _draw_coordinate_frame(ax, transform, axis_length, name):
        """在三维坐标轴中画出一个位姿坐标系。"""
        origin = transform[0:3, 3]
        rotation = transform[0:3, 0:3]
        colors = ["red", "green", "blue"]
        labels = ["X", "Y", "Z"]

        for index in range(3):
            direction = rotation[:, index]
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                direction[0],
                direction[1],
                direction[2],
                length=axis_length,
                color=colors[index],
                normalize=True,
            )

            end_point = origin + direction * axis_length
            ax.text(
                end_point[0],
                end_point[1],
                end_point[2],
                name + "-" + labels[index],
                color=colors[index],
            )

    def visualize_robot(self, theta):
        """画出机械臂连杆、基坐标系和末端执行器坐标系。"""
        matrices, t_final = self.fkine(theta)

        positions = [[0.0, 0.0, 0.0]]
        current_transform = np.eye(4)

        for matrix in matrices:
            current_transform = np.matmul(current_transform, matrix)
            current_position = current_transform[0:3, 3]
            positions.append(current_position.tolist())

        positions = np.array(positions)

        figure = plt.figure(figsize=(9, 7))
        ax = figure.add_subplot(111, projection="3d")

        ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color="black",
            linewidth=3,
        )
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color="orange",
            s=45,
        )

        self._draw_coordinate_frame(ax, np.eye(4), 120.0, "Base")
        self._draw_coordinate_frame(ax, t_final, 120.0, "Tool")

        pose = self.homography2pose(t_final)
        rpy_degree = np.degrees(pose[3:6])
        display_position = np.round(pose[0:3], 2).tolist()
        display_rpy = np.round(rpy_degree, 2).tolist()

        title = (
            "Tool xyz(mm): "
            + str(display_position)
            + "\nTool RPY(deg): "
            + str(display_rpy)
        )
        ax.set_title(title)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")

        minimum = np.min(positions, axis=0)
        maximum = np.max(positions, axis=0)
        center = (minimum + maximum) / 2.0
        radius = np.max(maximum - minimum) * 0.65

        if radius < 250.0:
            radius = 250.0

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect([1, 1, 1])
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        return t_final


def main():
    robot = RobotKinematics()
    joint_angles_degree = np.array([0, 70, -90, 0, -85, 0])
    joint_angles_radian = np.radians(joint_angles_degree)

    _, t_final = robot.fkine(joint_angles_radian)
    rotation = t_final[0:3, 0:3]
    rpy = robot._rotation2rpy(rotation)
    rebuilt_rotation = robot._rpy2rotation(rpy)
    pose = robot.homography2pose(t_final)
    rebuilt_transform = robot.pose2homography(pose)

    np.set_printoptions(precision=6, suppress=True)

    print("末端旋转矩阵:")
    print(rotation)
    print("\n末端 RPY 角(deg):")
    print(np.degrees(rpy))
    print("\n由 RPY 角重新得到的旋转矩阵:")
    print(rebuilt_rotation)
    print("\n旋转矩阵最大误差:", np.max(np.abs(rotation - rebuilt_rotation)))
    print("位姿矩阵最大误差:", np.max(np.abs(t_final - rebuilt_transform)))

    robot.visualize_robot(joint_angles_radian)


if __name__ == "__main__":
    main()
