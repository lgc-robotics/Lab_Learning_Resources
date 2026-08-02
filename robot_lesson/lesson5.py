"""Lesson 5：用微分法和矢量积法计算机械臂雅可比矩阵。"""

import numpy as np


class RobotKinematics:
    """RML63 六轴机械臂的正运动学与雅可比模型。"""

    def __init__(self, jacobian_method="differential", tool_length=140.0):
        self.tool_length = tool_length
        methods = {
            "differential": self._differential_jacobian,
            "vector": self._vector_jacobian,
        }
        if jacobian_method not in methods:
            raise ValueError(
                f"雅可比方法 '{jacobian_method}' 不存在，"
                "请选择 'differential' 或 'vector'。"
            )
        self.input_jacobian = methods[jacobian_method]

    def get_jacobian(self, q):
        """使用初始化时选定的方法计算雅可比矩阵。"""
        return self.input_jacobian(q)

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
        """返回六个相邻坐标系变换矩阵和末端位姿。"""
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

    def _differential_jacobian(self, q):
        """用矩阵显式微分公式计算 6×6 雅可比矩阵。"""
        A, _ = self.fkine(q)

        T56 = A[5]
        T46 = A[4] @ T56
        T36 = A[3] @ T46
        T26 = A[2] @ T36
        T16 = A[1] @ T26
        T06 = A[0] @ T16

        j11 = np.array([
            -T16[0, 0] * T16[1, 3] + T16[1, 0] * T16[0, 3],
            -T16[0, 1] * T16[1, 3] + T16[1, 1] * T16[0, 3],
            -T16[0, 2] * T16[1, 3] + T16[1, 2] * T16[0, 3],
            T16[2, 0],
            T16[2, 1],
            T16[2, 2],
        ])
        j22 = np.array([
            -T26[0, 0] * T26[1, 3] + T26[1, 0] * T26[0, 3],
            -T26[0, 1] * T26[1, 3] + T26[1, 1] * T26[0, 3],
            -T26[0, 2] * T26[1, 3] + T26[1, 2] * T26[0, 3],
            T26[2, 0],
            T26[2, 1],
            T26[2, 2],
        ])
        j33 = np.array([
            -T36[0, 0] * T36[1, 3] + T36[1, 0] * T36[0, 3],
            -T36[0, 1] * T36[1, 3] + T36[1, 1] * T36[0, 3],
            -T36[0, 2] * T36[1, 3] + T36[1, 2] * T36[0, 3],
            T36[2, 0],
            T36[2, 1],
            T36[2, 2],
        ])
        j44 = np.array([
            -T46[0, 0] * T46[1, 3] + T46[1, 0] * T46[0, 3],
            -T46[0, 1] * T46[1, 3] + T46[1, 1] * T46[0, 3],
            -T46[0, 2] * T46[1, 3] + T46[1, 2] * T46[0, 3],
            T46[2, 0],
            T46[2, 1],
            T46[2, 2],
        ])
        j55 = np.array([
            -T56[0, 0] * T56[1, 3] + T56[1, 0] * T56[0, 3],
            -T56[0, 1] * T56[1, 3] + T56[1, 1] * T56[0, 3],
            -T56[0, 2] * T56[1, 3] + T56[1, 2] * T56[0, 3],
            T56[2, 0],
            T56[2, 1],
            T56[2, 2],
        ])
        j66 = np.array([0, 0, 0, 0, 0, 1])

        rotation = T06[:3, :3]
        frame_transform = np.zeros((6, 6))
        frame_transform[:3, :3] = rotation
        frame_transform[3:, 3:] = rotation

        return frame_transform @ np.column_stack([
            j11,
            j22,
            j33,
            j44,
            j55,
            j66,
        ])

    def _vector_jacobian(self, q):
        """用 z_i × (p_end - p_i) 公式计算 6×6 雅可比矩阵。"""
        matrices, _ = self.fkine(q)

        transforms = []
        current = np.eye(4)
        for matrix in matrices:
            current = current @ matrix
            transforms.append(current)

        p_end = transforms[-1][:3, 3]
        jacobian = np.zeros((6, 6))
        for i, transform in enumerate(transforms):
            z_i = transform[:3, 2]
            p_i = transform[:3, 3]
            jacobian[:3, i] = np.cross(z_i, p_end - p_i)
            jacobian[3:, i] = z_i

        return jacobian


def main():
    q_deg = np.array([0, 70, -90, 0, -85, 0])
    q = np.radians(q_deg)

    differential_robot = RobotKinematics("differential")
    vector_robot = RobotKinematics("vector")
    J_differential = differential_robot.get_jacobian(q)
    J_vector = vector_robot.get_jacobian(q)

    np.set_printoptions(precision=4, suppress=True)
    print("关节角（度）：", q_deg)
    print(f"\n微分法雅可比：\n{J_differential}")
    print(f"\n矢量积法雅可比：\n{J_vector}")
    print("\n两种方法最大元素误差：", np.max(np.abs(J_differential - J_vector)))


if __name__ == "__main__":
    main()
