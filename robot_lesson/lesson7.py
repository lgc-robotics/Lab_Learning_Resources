"""Lesson 7: 基于高斯-牛顿法的机械臂逆运动学。
"""

import numpy as np


class RobotKinematics:
    """RML-63 六自由度机械臂运动学。"""

    def __init__(self, tool_length=140):
        self.q_min = np.radians([-178, -178, -178, -178, -178, -360])
        self.q_max = np.radians([178, 178, 145, 178, 178, 360])
        self.tool_length = tool_length
        self.input_jacobian = self._differential_jacobian

    def get_jacobian(self, q):
        return self.input_jacobian(q)

    @staticmethod
    def _DHTrans(alpha, a, d, theta):
        """根据改进 D-H 参数计算单个关节的齐次变换矩阵。"""
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
        """由 6 个关节角计算各节变换矩阵和末端齐次矩阵。"""
        initial_offset = np.array([0, -np.pi / 2, np.pi / 2, 0, np.pi, np.pi])
        th = np.array(theta) + initial_offset
        d = np.array([162.5, 0, 0, 405, 0, 132.3 + self.tool_length])
        a = np.array([0, -86, 380, 69, 0, 0])
        alpha = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, -np.pi / 2])

        matrices = []
        t_final = np.eye(4)
        for i in range(6):
            ai = self._DHTrans(alpha[i], a[i], d[i], th[i])
            matrices.append(ai)
            t_final = t_final @ ai

        return matrices, t_final

    def _differential_jacobian(self, q):
        """使用矩阵显式微分法计算 6x6 雅可比矩阵。"""
        a, _ = self.fkine(q)

        t56 = a[5]
        t46 = a[4] @ t56
        t36 = a[3] @ t46
        t26 = a[2] @ t36
        t16 = a[1] @ t26
        t06 = a[0] @ t16

        j11 = np.array([
            -t16[0, 0] * t16[1, 3] + t16[1, 0] * t16[0, 3],
            -t16[0, 1] * t16[1, 3] + t16[1, 1] * t16[0, 3],
            -t16[0, 2] * t16[1, 3] + t16[1, 2] * t16[0, 3],
            t16[2, 0],
            t16[2, 1],
            t16[2, 2],
        ])
        j22 = np.array([
            -t26[0, 0] * t26[1, 3] + t26[1, 0] * t26[0, 3],
            -t26[0, 1] * t26[1, 3] + t26[1, 1] * t26[0, 3],
            -t26[0, 2] * t26[1, 3] + t26[1, 2] * t26[0, 3],
            t26[2, 0],
            t26[2, 1],
            t26[2, 2],
        ])
        j33 = np.array([
            -t36[0, 0] * t36[1, 3] + t36[1, 0] * t36[0, 3],
            -t36[0, 1] * t36[1, 3] + t36[1, 1] * t36[0, 3],
            -t36[0, 2] * t36[1, 3] + t36[1, 2] * t36[0, 3],
            t36[2, 0],
            t36[2, 1],
            t36[2, 2],
        ])
        j44 = np.array([
            -t46[0, 0] * t46[1, 3] + t46[1, 0] * t46[0, 3],
            -t46[0, 1] * t46[1, 3] + t46[1, 1] * t46[0, 3],
            -t46[0, 2] * t46[1, 3] + t46[1, 2] * t46[0, 3],
            t46[2, 0],
            t46[2, 1],
            t46[2, 2],
        ])
        j55 = np.array([
            -t56[0, 0] * t56[1, 3] + t56[1, 0] * t56[0, 3],
            -t56[0, 1] * t56[1, 3] + t56[1, 1] * t56[0, 3],
            -t56[0, 2] * t56[1, 3] + t56[1, 2] * t56[0, 3],
            t56[2, 0],
            t56[2, 1],
            t56[2, 2],
        ])
        j66 = np.array([0, 0, 0, 0, 0, 1])

        rotation = t06[:3, :3]
        transform = np.zeros((6, 6))
        transform[:3, :3] = rotation
        transform[3:, 3:] = rotation

        return transform @ np.column_stack([j11, j22, j33, j44, j55, j66])

    @staticmethod
    def _pose_error(t_curr, t_target):
        """计算当前位姿相对目标位姿的 6 维位置与轴角误差。"""
        p_err = t_curr[:3, 3] - t_target[:3, 3]

        r_curr = t_curr[:3, :3]
        r_target = t_target[:3, :3]
        r_err = r_curr @ r_target.T

        cos_theta = np.clip((np.trace(r_err) - 1) / 2.0, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        if np.abs(theta) < 1e-7:
            w_err = np.zeros(3)
        else:
            axis = (1 / (2 * np.sin(theta))) * np.array([
                r_err[2, 1] - r_err[1, 2],
                r_err[0, 2] - r_err[2, 0],
                r_err[1, 0] - r_err[0, 1],
            ])
            w_err = theta * axis

        return np.hstack((p_err, w_err))

    @staticmethod
    def _rotation2rpy(rotation):
        """将旋转矩阵转换为 ZYX 顺序的 RPY 欧拉角。"""
        if rotation.shape == (4, 4):
            rotation = rotation[:3, :3]

        pitch = np.arctan2(
            -rotation[2, 0],
            np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2),
        )
        if np.abs(np.cos(pitch)) < 1e-10:
            yaw = 0.0
            roll = np.arctan2(np.sign(pitch) * rotation[0, 1], rotation[1, 1])
        else:
            roll = np.arctan2(rotation[2, 1], rotation[2, 2])
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])

        return np.array([roll, pitch, yaw])

    @staticmethod
    def _rpy2rotation(euler):
        """将 RPY 欧拉角转换为 ZYX 顺序的旋转矩阵。"""
        rx = np.array([
            [1, 0, 0],
            [0, np.cos(euler[0]), -np.sin(euler[0])],
            [0, np.sin(euler[0]), np.cos(euler[0])],
        ])
        ry = np.array([
            [np.cos(euler[1]), 0, np.sin(euler[1])],
            [0, 1, 0],
            [-np.sin(euler[1]), 0, np.cos(euler[1])],
        ])
        rz = np.array([
            [np.cos(euler[2]), -np.sin(euler[2]), 0],
            [np.sin(euler[2]), np.cos(euler[2]), 0],
            [0, 0, 1],
        ])
        return rz @ ry @ rx

    def homography2pose(self, homography):
        """将 4x4 齐次矩阵转换为 [x, y, z, rx, ry, rz]。"""
        rx, ry, rz = self._rotation2rpy(homography)
        return np.array([
            homography[0, 3],
            homography[1, 3],
            homography[2, 3],
            rx,
            ry,
            rz,
        ])

    def pose2homography(self, pose):
        """将 [x, y, z, rx, ry, rz] 转换为 4x4 齐次矩阵。"""
        x, y, z, rx, ry, rz = pose
        homography = np.eye(4)
        homography[:3, :3] = self._rpy2rotation(np.array([rx, ry, rz]))
        homography[:3, 3] = np.array([x, y, z])
        return homography

    def gn_ik(
        self,
        target_pose,
        initial_q,
        max_iter=100,
        tol=1e-7,
        attempts=30,
        record_trace=None,
    ):
        """使用高斯-牛顿法求逆运动学解。

        核心公式：dq = -(J.T @ J)^(-1) @ J.T @ error。
        返回字典中的关节角单位为弧度。
        """
        t_des = self.pose2homography(target_pose)
        if initial_q is not None:
            initial_q = np.array(initial_q, float)

        best_result = None

        for n in range(attempts):
            if initial_q is not None and n == 0:
                q = initial_q.copy()
            else:
                q = np.random.uniform(self.q_min, self.q_max)
            ok = True

            for i in range(max_iter):
                _, t_curr = self.fkine(q)
                error = self._pose_error(t_curr, t_des)

                if np.linalg.norm(error) < tol:
                    break

                j = self.get_jacobian(q)
                h = j.T @ j
                g = j.T @ error

                try:
                    dq = -np.linalg.solve(h, g)
                except np.linalg.LinAlgError:
                    dq = -np.linalg.pinv(j) @ error

                q = q + 0.01 * dq

            _, t_final = self.fkine(q)
            final_error = np.linalg.norm(self._pose_error(t_final, t_des))
            success = final_error < tol and ok

            result = {
                "q": q.copy(),
                "success": success,
                "error": final_error,
                "iterations": i if final_error < tol else max_iter,
            }

            if best_result is None or result["error"] < best_result["error"]:
                best_result = result

            if result["success"]:
                return result


def main():
    """从邻近初值求解一个由正运动学生成的可达目标。"""
    robot = RobotKinematics()
    target_q = np.deg2rad([10, -20, 30, 15, -25, 40])
    initial_q = target_q + np.deg2rad([2, -2, 2, -2, 2, -2])

    _, target_t = robot.fkine(target_q)
    target_pose = robot.homography2pose(target_t)
    result = robot.gn_ik(
        target_pose,
        initial_q=initial_q,
        max_iter=2500,
        tol=1e-7,
        attempts=1,
    )

    print("目标关节角(度):", np.round(np.rad2deg(target_q), 4))
    print("初始关节角(度):", np.round(np.rad2deg(initial_q), 4))
    if result is None:
        print("高斯-牛顿法未在指定迭代次数内收敛。")
        return

    print("求解关节角(度):", np.round(np.rad2deg(result["q"]), 4))
    print("是否收敛:", result["success"])
    print("最终位姿误差:", f"{result['error']:.3e}")
    print("迭代次数:", result["iterations"])


if __name__ == "__main__":
    main()
