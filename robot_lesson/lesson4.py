"""Lesson 4：六轴机械臂解析法逆解及正运动学回代。"""

import math

import numpy as np


# 机械臂的改进 D-H 参数

# 长度单位都是 mm。
DH_A = [0.0, 86.0, 380.0, -69.0, 0.0, 0.0]
DH_D = [172.0, 0.0, -10.2, 405.0, 0.0, 115.0]

DH_ALPHA_DEGREE = [0.0, 90.0, 0.0, 90.0, -90.0, -90.0]
JOINT_OFFSET_DEGREE = [-90.0, 90.0, 90.0, 0.0, 180.0, 90.0]

# 程序内部的三角函数全部使用弧度。
DH_ALPHA = []
JOINT_OFFSET = []

for value in DH_ALPHA_DEGREE:
    DH_ALPHA.append(math.radians(value))

for value in JOINT_OFFSET_DEGREE:
    JOINT_OFFSET.append(math.radians(value))


def normalize_radian(angle):
    """把一个弧度角整理到 [-pi, pi)。"""
    while angle >= math.pi:
        angle = angle - 2.0 * math.pi

    while angle < -math.pi:
        angle = angle + 2.0 * math.pi

    return angle


def normalize_degree(angle):
    """把一个角度整理到 [-180, 180)。"""
    while angle >= 180.0:
        angle = angle - 360.0

    while angle < -180.0:
        angle = angle + 360.0

    return angle


def safe_sqrt(value):
    """带少量数值保护的平方根；确实小于零时返回 None。"""
    small_number = 1e-9

    if value < -small_number:
        return None

    if value < 0.0:
        value = 0.0

    return math.sqrt(value)


def _DHTrans(alpha, a, d, theta):
    """按照本机械臂使用的改进 D-H 公式计算一个 A 矩阵。"""
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)

    matrix = np.array([
        [cos_theta, -sin_theta, 0.0, a],
        [
            sin_theta * cos_alpha,
            cos_theta * cos_alpha,
            -sin_alpha,
            -sin_alpha * d,
        ],
        [
            sin_theta * sin_alpha,
            cos_theta * sin_alpha,
            cos_alpha,
            cos_alpha * d,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ])

    return matrix


# 正运动学

def fkine(theta):
    """输入六个关节角（弧度），返回各节矩阵和末端矩阵。"""
    if len(theta) != 6:
        raise ValueError("必须输入 6 个关节角。")

    th = []

    for index in range(6):
        current_theta = theta[index] + JOINT_OFFSET[index]
        th.append(current_theta)

    matrices = []
    t_final = np.eye(4)

    for index in range(6):
        current_matrix = _DHTrans(
            DH_ALPHA[index],
            DH_A[index],
            DH_D[index],
            th[index],
        )
        matrices.append(current_matrix)
        t_final = np.matmul(t_final, current_matrix)

    return matrices, t_final


def calculate_t16(target_matrix, theta1):
    """按照 T16 = A1 的逆矩阵 * T06 分离第一个关节。"""
    matrix_a1 = _DHTrans(
        DH_ALPHA[0], DH_A[0], DH_D[0], theta1
    )
    inverse_a1 = np.linalg.inv(matrix_a1)
    transform_16 = np.matmul(inverse_a1, target_matrix)

    return transform_16

# 解析求 theta1
def solve_theta1(target_matrix):
    """根据总变换矩阵的位置列和第三姿态列求 theta1。"""
    px = target_matrix[0, 3]
    py = target_matrix[1, 3]
    ax = target_matrix[0, 2]
    ay = target_matrix[1, 2]

    d3 = DH_D[2]
    d6 = DH_D[5]

    # 由 T06 的元素展开式移项得到：
    # N = px - d6 * ax
    # M = py - d6 * ay
    # N * sin(theta1) - M * cos(theta1) = d3
    value_n = px - d6 * ax
    value_m = py - d6 * ay

    square_value = value_n * value_n + value_m * value_m - d3 * d3
    square_root = safe_sqrt(square_value)

    theta1_list = []

    if square_root is None:
        return theta1_list

    common_angle = math.atan2(value_m, value_n)

    theta1_first = math.atan2(d3, square_root) + common_angle
    theta1_second = math.atan2(d3, -square_root) + common_angle

    theta1_first = normalize_radian(theta1_first)
    theta1_second = normalize_radian(theta1_second)

    theta1_list.append(theta1_first)

    difference = normalize_radian(theta1_second - theta1_first)
    if abs(difference) > 1e-8:
        theta1_list.append(theta1_second)

    return theta1_list

# 由 theta1 得到位置方程中的 X、Y
def calculate_x_y(transform_16):
    """比较 T16 的矩阵元素，得到后面要用的 X、Y。"""
    a2 = DH_A[1]
    d6 = DH_D[5]

    # T16 = A2*A3*A4*A5*A6。
    # 直接比较 T16 的 (1,4)、(1,3)、(3,4)、(3,3) 元素并移项。
    value_x = transform_16[0, 3] - a2 - d6 * transform_16[0, 2]
    value_y = transform_16[2, 3] - d6 * transform_16[2, 2]

    return value_x, value_y


# 解析求 theta3
def solve_theta3(value_x, value_y):
    """根据 X、Y 的平方和方程求 theta3 的两组可能值。"""
    a3 = DH_A[2]
    a4 = DH_A[3]
    d4 = DH_D[3]

    # 由 X^2 + Y^2 展开并整理得到：
    # D = a4*cos(theta3) + d4*sin(theta3)
    value_d = (
        value_x * value_x
        + value_y * value_y
        - a3 * a3
        - a4 * a4
        - d4 * d4
    ) / (2.0 * a3)

    square_value = a4 * a4 + d4 * d4 - value_d * value_d
    square_root = safe_sqrt(square_value)

    theta3_list = []

    if square_root is None:
        return theta3_list

    fixed_angle = math.atan2(a4, d4)

    theta3_first = math.atan2(value_d, square_root) - fixed_angle
    theta3_second = math.atan2(value_d, -square_root) - fixed_angle

    theta3_first = normalize_radian(theta3_first)
    theta3_second = normalize_radian(theta3_second)

    theta3_list.append(theta3_first)

    difference = normalize_radian(theta3_second - theta3_first)
    if abs(difference) > 1e-8:
        theta3_list.append(theta3_second)

    return theta3_list


# 解析求 theta2

def solve_theta2(value_x, value_y, theta3):
    """把已经求出的 theta3 代回 X、Y 方程，求 theta2。"""
    a3 = DH_A[2]
    a4 = DH_A[3]
    d4 = DH_D[3]

    # 把 X、Y 直接整理成 theta2 的形式：
    # X = A*cos(theta2) - B*sin(theta2)
    # Y = A*sin(theta2) + B*cos(theta2)
    # 其中 A、B 的正负号必须由 theta3 自己决定，不能沿用
    # 求 theta3 时平方根的正负号。这正是旧代码会产生假解的地方。
    value_a = (
        a3
        + a4 * math.cos(theta3)
        + d4 * math.sin(theta3)
    )
    value_b = a4 * math.sin(theta3) - d4 * math.cos(theta3)

    theta2 = math.atan2(value_y, value_x) - math.atan2(value_b, value_a)
    theta2 = normalize_radian(theta2)

    return theta2


# 继续逐级左乘逆矩阵，求 theta4、theta5、theta6

def calculate_t36(transform_16, theta2, theta3):
    """依次计算 T26=A2^-1*T16、T36=A3^-1*T26。"""
    matrix_a2 = _DHTrans(
        DH_ALPHA[1], DH_A[1], DH_D[1], theta2
    )
    inverse_a2 = np.linalg.inv(matrix_a2)
    transform_26 = np.matmul(inverse_a2, transform_16)

    matrix_a3 = _DHTrans(
        DH_ALPHA[2], DH_A[2], DH_D[2], theta3
    )
    inverse_a3 = np.linalg.inv(matrix_a3)
    transform_36 = np.matmul(inverse_a3, transform_26)

    return transform_36


def solve_theta5_theta6(transform_36, theta4):
    """已知 theta4 后，由 T46、T56 分别求 theta5、theta6。"""
    matrix_a4 = _DHTrans(
        DH_ALPHA[3], DH_A[3], DH_D[3], theta4
    )
    inverse_a4 = np.linalg.inv(matrix_a4)
    transform_46 = np.matmul(inverse_a4, transform_36)

    # T46 = A5*A6。比较 (1,3)、(3,3) 元素：
    # T46[0,2] = -sin(theta5)
    # T46[2,2] = -cos(theta5)
    theta5 = math.atan2(-transform_46[0, 2], -transform_46[2, 2])
    theta5 = normalize_radian(theta5)

    matrix_a5 = _DHTrans(
        DH_ALPHA[4], DH_A[4], DH_D[4], theta5
    )
    inverse_a5 = np.linalg.inv(matrix_a5)
    transform_56 = np.matmul(inverse_a5, transform_46)

    # T56 就是 A6。比较第一行前两个元素：
    # T56[0,0] = cos(theta6)
    # T56[0,1] = -sin(theta6)
    theta6 = math.atan2(-transform_56[0, 1], transform_56[0, 0])
    theta6 = normalize_radian(theta6)

    return theta5, theta6


def solve_theta4_theta5_theta6(transform_36):
    """先由 T36 求 theta4，再逐级分离 theta5、theta6。"""
    g13 = transform_36[0, 2]
    g33 = transform_36[2, 2]

    # T36[0,2] = -cos(theta4)*sin(theta5)
    # T36[2,2] = -sin(theta4)*sin(theta5)
    sin_theta5_absolute = math.sqrt(g13 * g13 + g33 * g33)

    theta4_list = []

    if sin_theta5_absolute > 1e-8:
        theta4_first = math.atan2(-g33, -g13)
        theta4_second = normalize_radian(theta4_first + math.pi)

        theta4_list.append(normalize_radian(theta4_first))
        theta4_list.append(theta4_second)
    else:
        # 姿态奇异时 theta4、theta6 互相耦合，有无穷多组表示。
        # 固定 theta4=0，只求一组代表解，避免人为制造重复结果。
        theta4_list.append(0.0)

    result = []

    for theta4 in theta4_list:
        theta5, theta6 = solve_theta5_theta6(transform_36, theta4)
        one_result = [theta4, theta5, theta6]
        result.append(one_result)

    return result


def dh_angles_to_joint_degrees(dh_angles):
    """减去零位偏置，并把结果转换为角度。"""
    joint_degrees = []

    for index in range(6):
        joint_radian = dh_angles[index] - JOINT_OFFSET[index]
        joint_radian = normalize_radian(joint_radian)
        joint_degree = math.degrees(joint_radian)
        joint_degree = normalize_degree(joint_degree)
        joint_degrees.append(joint_degree)

    return joint_degrees


def solutions_are_same(first_solution, second_solution, tolerance=1e-5):
    """判断两组角度是否只相差整圈。"""
    for index in range(6):
        difference = first_solution[index] - second_solution[index]
        difference = normalize_degree(difference)

        if abs(difference) > tolerance:
            return False

    return True


def remove_duplicate_solutions(solution_list):
    """删除相差整圈的重复关节解。"""
    unique_solutions = []

    for current_solution in solution_list:
        already_exists = False

        for saved_solution in unique_solutions:
            if solutions_are_same(current_solution, saved_solution):
                already_exists = True
                break

        if already_exists is False:
            unique_solutions.append(current_solution)

    return unique_solutions


#  完整解析逆解主函数

def inverse_kinematics(target_matrix):
    """按照 theta1 -> theta3 -> theta2 -> theta4/5/6 的顺序求解。"""
    verified_solutions = []

    theta1_list = solve_theta1(target_matrix)

    for theta1 in theta1_list:
        transform_16 = calculate_t16(target_matrix, theta1)
        value_x, value_y = calculate_x_y(transform_16)
        theta3_list = solve_theta3(value_x, value_y)

        # 某一个 theta1 分支没有 theta3 解时，只跳过当前分支。
        # 不能像旧代码那样直接把其他分支也全部丢掉。
        for theta3 in theta3_list:
            theta2 = solve_theta2(value_x, value_y, theta3)
            transform_36 = calculate_t36(transform_16, theta2, theta3)
            last_three_list = solve_theta4_theta5_theta6(transform_36)

            for last_three in last_three_list:
                theta4 = last_three[0]
                theta5 = last_three[1]
                theta6 = last_three[2]

                dh_angles = [
                    theta1,
                    theta2,
                    theta3,
                    theta4,
                    theta5,
                    theta6,
                ]

                joint_solution = dh_angles_to_joint_degrees(dh_angles)

                # 每一组解析结果都必须通过正运动学回代。
                _, check_matrix = fkine(np.radians(joint_solution))

                if np.allclose(target_matrix, check_matrix, atol=1e-6):
                    verified_solutions.append(joint_solution)

    unique_solutions = remove_duplicate_solutions(verified_solutions)

    return unique_solutions


def main():
    np.random.seed(12345)
    test_count = 1000
    success_count = 0

    for test_index in range(test_count):
        joint_angles_degree = []

        for joint_index in range(6):
            random_angle = np.random.uniform(-165.0, 165.0)
            joint_angles_degree.append(random_angle)

        joint_angles_radian = np.radians(joint_angles_degree)
        _, target_matrix = fkine(joint_angles_radian)
        solutions = inverse_kinematics(target_matrix)

        original_solution_was_found = False
        all_solutions_are_correct = True

        for solution in solutions:
            _, check_matrix = fkine(np.radians(solution))

            if np.allclose(target_matrix, check_matrix, atol=1e-6) is False:
                all_solutions_are_correct = False

            if solutions_are_same(solution, joint_angles_degree, tolerance=1e-4):
                original_solution_was_found = True

        success = original_solution_was_found and all_solutions_are_correct

        if success:
            success_count = success_count + 1
            result_text = "成功"
        else:
            result_text = "失败"

        print(
            "第",
            test_index + 1,
            "次测试:",
            result_text,
            "，解析解数量:",
            len(solutions),
        )

    success_rate = success_count / test_count * 100.0

    print("\n测试成功:", success_count, "/", test_count)
    print("成功率:", "{:.2f}%".format(success_rate))


if __name__ == "__main__":
    main()
