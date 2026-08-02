"""Lesson 8: 基于 Levenberg-Marquardt 算法的机械臂逆运动学。
"""

import numpy as np


class RobotKinematics:
    """RML-63 六自由度机械臂运动学。"""

    def __init__(self, tool_length=140):
        self.q_min = np.radians([-178, -178, -178, -178, -178, -360])
        self.q_max = np.radians([178, 178, 145, 178, 178, 360])
        self.tool_length = tool_length

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

    def _vector_jacobian(self, q):
        """使用关节轴与末端位置的矢量积计算 6x6 雅可比矩阵。"""
        matrices, _ = self.fkine(q)
        a1, a2, a3, a4, a5, a6 = matrices

        t01 = a1
        t02 = t01 @ a2
        t03 = t02 @ a3
        t04 = t03 @ a4
        t05 = t04 @ a5
        t06 = t05 @ a6

        p_end = t06[:3, 3]
        z1, p1 = t01[:3, 2], t01[:3, 3]
        z2, p2 = t02[:3, 2], t02[:3, 3]
        z3, p3 = t03[:3, 2], t03[:3, 3]
        z4, p4 = t04[:3, 2], t04[:3, 3]
        z5, p5 = t05[:3, 2], t05[:3, 3]
        z6, p6 = t06[:3, 2], t06[:3, 3]

        jacobian = np.zeros((6, 6))
        jacobian[:3, 0] = np.cross(z1, p_end - p1)
        jacobian[3:, 0] = z1
        jacobian[:3, 1] = np.cross(z2, p_end - p2)
        jacobian[3:, 1] = z2
        jacobian[:3, 2] = np.cross(z3, p_end - p3)
        jacobian[3:, 2] = z3
        jacobian[:3, 3] = np.cross(z4, p_end - p4)
        jacobian[3:, 3] = z4
        jacobian[:3, 4] = np.cross(z5, p_end - p5)
        jacobian[3:, 4] = z5
        jacobian[:3, 5] = np.cross(z6, p_end - p6)
        jacobian[3:, 5] = z6
        return jacobian

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

    def lm_ik(
        self,
        target_pose,
        initial_q=None,
        max_iter=400,
        lam=0.1,
        rlimit=100,
        tol=1e-3,
    ):
        """使用 LM 法求逆运动学解（原代码注释：91% 成功率）。"""
        success = False
        iterations = 0
        rejcount = 0

        if initial_q is None:
            q = np.random.uniform(self.q_min, self.q_max)
        else:
            q = np.array(initial_q, dtype=float).copy()

        t_target = self.pose2homography(target_pose)

        for iters in range(max_iter):
            iterations = iters + 1

            _, t_start = self.fkine(q)
            err = self._pose_error(t_start, t_target)
            err = err[:, np.newaxis]
            if np.linalg.norm(err) < tol:
                success = True
                break

            j = self._vector_jacobian(q)
            jtj = j.T @ j
            dq = (
                -np.linalg.inv(jtj + (lam + 1e-8) * np.eye(jtj.shape[0]))
                @ j.T
                @ err
            )
            q_new = q + dq.squeeze()

            _, t_curr = self.fkine(q_new)
            err_new = self._pose_error(t_curr, t_target)
            err_new_norm = np.linalg.norm(err_new)
            err_norm = np.linalg.norm(err)

            if err_new_norm < err_norm:
                q = q_new
                lam = lam / 2
                rejcount = 0
            else:
                lam = lam * 2
                rejcount += 1
                if rejcount > rlimit:
                    break

            q = (q + np.pi) % (2 * np.pi) - np.pi

        _, t_final = self.fkine(q)
        final_error = np.linalg.norm(self._pose_error(t_final, t_target))
        if final_error < tol:
            success = True

        return q, success


def main():
    """从邻近初值求解一个由正运动学生成的可达目标。"""
    robot = RobotKinematics()
    target_q = np.deg2rad([10, -20, 30, 15, -25, 40])
    initial_q = target_q + np.deg2rad([8, -8, 8, -8, 8, -8])

    _, target_t = robot.fkine(target_q)
    target_pose = robot.homography2pose(target_t)
    solution_q, success = robot.lm_ik(target_pose, initial_q=initial_q)

    _, solution_t = robot.fkine(solution_q)
    final_error = np.linalg.norm(robot._pose_error(solution_t, target_t))

    print("目标关节角(度):", np.round(np.rad2deg(target_q), 4))
    print("初始关节角(度):", np.round(np.rad2deg(initial_q), 4))
    print("求解关节角(度):", np.round(np.rad2deg(solution_q), 4))
    print("是否收敛:", success)
    print("最终位姿误差:", f"{final_error:.3e}")


if __name__ == "__main__":
    main()
