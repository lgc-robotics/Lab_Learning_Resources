"""Lesson 1：六轴机械臂正运动学。"""

import numpy as np


class RobotKinematics:
    """RML63 六轴机械臂的改进 D-H 正运动学模型。"""

    def __init__(self, tool_length=220.0):
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
        """
        计算六轴机械臂正运动学。

        参数 theta：六个关节角，单位为弧度。
        返回 (matrices, T_final)：六个相邻坐标系变换矩阵和末端位姿。
        """
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
        T_final = np.eye(4)
        for i in range(6):
            Ai = self._DHTrans(alpha[i], a[i], d[i], th[i])
            matrices.append(Ai)
            T_final = T_final @ Ai

        return matrices, T_final


def main():
    robot = RobotKinematics()
    joint_angles_deg = np.array([0, 70, -90, 0, -85, 0])
    matrices, T_final = robot.fkine(np.radians(joint_angles_deg))

    np.set_printoptions(precision=4, suppress=True)
    print("关节角（度）：", joint_angles_deg)
    for i, matrix in enumerate(matrices, start=1):
        print(f"\nA{i} =\n{matrix}")
    print(f"\n末端位姿 T06 =\n{T_final}")


if __name__ == "__main__":
    main()
