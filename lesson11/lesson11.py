"""Lesson 11：罚函数-高斯牛顿法机械臂逆运动学。
"""

import numpy as np
import matplotlib.pyplot as plt


# MDH 参数，长度单位：mm
END_EFF_LEN = 200
DH_A = [0, -86, 380, 69, 0, 0]
DH_ALPHA = [0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, -np.pi / 2]
DH_D = [162.5, 0, 0, 405, 0, 132.3 + END_EFF_LEN]
DH_OFFSET = [0, -np.pi / 2, np.pi / 2, 0, np.pi, np.pi]


class Robot_6:
    def cal_link_matrix(self, alpha, a, d, theta):
        """计算单个关节的齐次变换矩阵。"""
        hx = np.array([
            [1, 0, 0, a],
            [0, np.cos(alpha), -np.sin(alpha), 0],
            [0, np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 0, 1],
        ])

        hz = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, d],
            [0, 0, 0, 1],
        ])

        return np.matmul(hx, hz)

    def fkine(self, thetas):
        """正运动学。输入为各关节角（弧度），输出 4x4 位姿矩阵。"""
        for i in range(len(thetas)):
            Ti = self.cal_link_matrix(
                DH_ALPHA[i], DH_A[i], DH_D[i], DH_OFFSET[i] + thetas[i]
            )
            if i == 0:
                T = Ti
            else:
                T = np.matmul(T, Ti)

        return T

    def vex(self, S):
        """从反对称矩阵中取出旋转轴。"""
        W = np.zeros((3, 1), dtype=np.float64)
        W[0, 0] = 0.5 * (S[2, 1] - S[1, 2])
        W[1, 0] = 0.5 * (S[0, 2] - S[2, 0])
        W[2, 0] = 0.5 * (S[1, 0] - S[0, 1])

        return W

    def tr2delta(self, T0, T1):
        """计算目标位姿 T0 与当前位姿 T1 的平移、旋转误差。"""
        delta = np.zeros((6, 1))
        R0 = T0[:3, :3]
        R1 = T1[:3, :3]

        dR = self.vex(np.matmul(R0, R1.T))
        t = np.array([
            T0[0, 3] - T1[0, 3],
            T0[1, 3] - T1[1, 3],
            T0[2, 3] - T1[2, 3],
        ])
        t = t.reshape(3, 1)

        for r in range(3):
            delta[r, 0] = t[r, 0]
            delta[r + 3, 0] = dR[r, 0]

        return delta

    def myJacobian(self, theta):
        """用微分变换法计算 6x6 雅可比矩阵。"""
        n_dof = len(theta)
        Jn = np.zeros((6, len(theta)))

        T_n_i = np.eye(4)

        for i in range(len(theta) - 1, 0, -1):
            Ti = self.cal_link_matrix(
                DH_ALPHA[i], DH_A[i], DH_D[i], DH_OFFSET[i] + theta[i]
            )
            T_n_i = np.dot(Ti, T_n_i)

            R = T_n_i[0:3, 0:3]
            p = T_n_i[:3, 3:4]

            n = R[:, 0].reshape(3, 1)
            o = R[:, 1].reshape(3, 1)
            a = R[:, 2].reshape(3, 1)

            pn = np.cross(p.T, n.T).T
            po = np.cross(p.T, o.T).T
            pa = np.cross(p.T, a.T).T

            j = i - 1
            Jn[0, j] = pn[2, 0]
            Jn[1, j] = po[2, 0]
            Jn[2, j] = pa[2, 0]
            Jn[3, j] = n[2, 0]
            Jn[4, j] = o[2, 0]
            Jn[5, j] = a[2, 0]

        Jn[5, n_dof - 1] = 1

        T06 = self.fkine(theta)
        R06 = T06[0:3, 0:3]
        T = np.zeros((6, 6))

        for r in range(3):
            for c in range(3):
                T[r, c] = R06[r, c]
                T[r + 3, c + 3] = R06[r, c]

        J = np.dot(T, Jn)

        return J

    def penalty_grad_2(self, theta):
        """计算第三关节越界时的罚函数梯度。"""
        if theta > np.deg2rad(144):
            return np.deg2rad(144) - theta
        elif theta < np.deg2rad(-177):
            return theta + np.deg2rad(177)
        else:
            return 0

    def IK_GN(self, T, theta_initial_value, is_plot=False):
        """使用带外点罚函数的高斯-牛顿法求逆解。"""
        theta = theta_initial_value
        iter_count = 0
        error = np.inf
        errors = []
        max_iter = 500
        tol = 1e-6
        lr = 0.1
        c_c = 1
        max_iterations = 45
        iteration = 0
        qlim = 0
        qlim_max = 5

        while iter_count < max_iter and error > tol:
            T_current = self.fkine(theta)
            e = self.tr2delta(T, T_current)

            J = self.myJacobian(theta)
            Jc = np.zeros((6, 6))

            if np.deg2rad(-177) < theta[2] < np.deg2rad(144):
                delta_theta = -np.linalg.pinv(J.T @ J) @ J.T @ e
                theta = theta - lr * delta_theta.squeeze(1)

                for i in range(len(theta)):
                    while theta[i] > np.pi:
                        theta[i] -= 2 * np.pi
                    while theta[i] < -np.pi:
                        theta[i] += 2 * np.pi

            while (
                not (np.deg2rad(-177) < theta[2] < np.deg2rad(144))
                and iteration <= max_iterations
            ):
                for i in range(len(theta)):
                    theta[i] = np.angle(np.exp(1j * theta[i]))

                p_grad = np.zeros(6).reshape((6, 1))
                p_grad[2] = self.penalty_grad_2(theta[2])

                if theta[2] < np.deg2rad(-177):
                    Jc[3, 3] = 1
                elif theta[2] > np.deg2rad(144):
                    Jc[3, 3] = -1

                c_c_float = float(c_c)
                delta_theta = -np.linalg.pinv(
                    (J.T @ J) + (Jc.T @ Jc) * c_c_float
                ) @ (J.T @ e + (Jc.T @ p_grad) * c_c_float)

                theta = theta + lr * delta_theta.squeeze(1)
                for i in range(len(theta)):
                    while theta[i] > np.pi:
                        theta[i] -= 2 * np.pi
                    while theta[i] < -np.pi:
                        theta[i] += 2 * np.pi

                c_c = 10 ** iteration

                if np.deg2rad(-177) < theta[2] < np.deg2rad(144):
                    iteration = 0
                else:
                    iteration += 1

                if iteration == max_iterations and qlim < qlim_max:
                    qlim += 1
                    iter_count = 0
                    iteration = 0
                    theta = np.random.uniform(
                        low=-np.pi / 2, high=np.pi / 2, size=(6,)
                    )

            for i in range(len(theta)):
                while theta[i] > np.pi:
                    theta[i] -= 2 * np.pi
                while theta[i] < -np.pi:
                    theta[i] += 2 * np.pi

            error = np.linalg.norm(e)
            errors.append(error)
            iter_count += 1

            if iter_count == max_iter and qlim < qlim_max:
                qlim += 1
                iter_count = 0
                theta = np.random.uniform(
                    low=-np.pi / 2, high=np.pi / 2, size=(6,)
                )

        if is_plot:
            plt.figure()
            plt.plot(errors)
            plt.xlabel("迭代次数")
            plt.ylabel("位姿误差")
            plt.show()

        return theta


def main():
    robot = Robot_6()

    target_theta = np.deg2rad([10, -20, 30, 10, 20, -15])
    target_pose = robot.fkine(target_theta)
    solution = robot.IK_GN(target_pose, np.zeros(6))
    final_error = np.linalg.norm(robot.tr2delta(target_pose, robot.fkine(solution)))

    print("目标位姿：")
    print(target_pose)
    print("\n逆解结果（度）：")
    print(np.rad2deg(solution))
    print(f"\n最终位姿误差：{final_error:.6e}")


if __name__ == "__main__":
    main()
