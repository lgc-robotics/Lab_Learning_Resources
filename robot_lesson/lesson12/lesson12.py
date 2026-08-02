"""Lesson 12: Pb_Init_、Pb_Func 及其完整运动学依赖。"""

import os
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data


LESSON_DIR = Path(__file__).resolve().parent
PYBULLET_ASSET_DIR = LESSON_DIR / "assets"


class Pb_Config:
    """PyBullet 场景和轨迹调试参数。"""

    base_dir = str(LESSON_DIR)
    pybullet_asset_dir = str(PYBULLET_ASSET_DIR)
    pybullet_car_stl_path = str(PYBULLET_ASSET_DIR / "car.STL")
    pybullet_robot_urdf_file = str(PYBULLET_ASSET_DIR / "RML63.urdf")

    arm_pos = [0.0, 600.0, 787.7]
    arm_rot = [0.0, 0.0, -np.pi / 2]

    tool_length = 140.0
    tool_radius = 20.0
    axis_length = 150.0

    camera_distance = 1.5
    camera_yaw = 80
    camera_pitch = -30
    camera_target = [0.0, 600.0, 1300.0]

    joint_indices = [0, 1, 2, 3, 4, 5]

    car_color = [0.7, 0.7, 0.7, 1.0]
    tool_color = [1.0, 0.0, 0.0, 0.6]

    preview_line_color = [0.0, 1.0, 0.0]
    preview_line_width = 2
    preview_point_size = 12
    progress_inner_color = [0.0, 1.0, 0.0]
    progress_inner_width = 3
    debug_interp_steps = 0
    debug_interp_dt = 0.01

    pose_text_anchor_offset = [-500.0, 0.0, 1200.0]
    pose_text_gap = 100.0
    auto_wait = 0.25


class RobotKinematics:
    """`pb_visualize.py` 实际需要的 RML63 运动学方法。"""

    def __init__(self, tool_length=140):
        self.q_min = np.radians([-178, -178, -178, -178, -178, -360])
        self.q_max = np.radians([178, 178, 145, 178, 178, 360])
        self.tool_length = tool_length

    @staticmethod
    def _DHTrans(alpha, a, d, theta):
        return np.array(
            [
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
            ]
        )

    def fkine(self, theta):
        """由六个关节角计算各节变换矩阵及末端齐次矩阵。"""
        theta = np.array(theta)
        initial_offset = np.array(
            [0, -np.pi / 2, np.pi / 2, 0, np.pi, np.pi]
        )
        th = theta + initial_offset
        d = np.array([162.5, 0, 0, 405, 0, 132.3 + self.tool_length])
        a = np.array([0, -86, 380, 69, 0, 0])
        alpha = np.array(
            [0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, -np.pi / 2]
        )

        matrices = []
        final_transform = np.eye(4)
        for index in range(6):
            transform = self._DHTrans(
                alpha[index],
                a[index],
                d[index],
                th[index],
            )
            matrices.append(transform)
            final_transform = final_transform @ transform
        return matrices, final_transform

    def get_jacobian(self, q):
        return self._differential_jacobian(q)

    def _differential_jacobian(self, q):
        """按原 `robot.py` 的微分法计算雅可比矩阵。"""
        transforms, _ = self.fkine(q)
        transform_56 = transforms[5]
        transform_46 = transforms[4] @ transform_56
        transform_36 = transforms[3] @ transform_46
        transform_26 = transforms[2] @ transform_36
        transform_16 = transforms[1] @ transform_26
        transform_06 = transforms[0] @ transform_16

        column_1 = np.array(
            [
                -transform_16[0, 0] * transform_16[1, 3]
                + transform_16[1, 0] * transform_16[0, 3],
                -transform_16[0, 1] * transform_16[1, 3]
                + transform_16[1, 1] * transform_16[0, 3],
                -transform_16[0, 2] * transform_16[1, 3]
                + transform_16[1, 2] * transform_16[0, 3],
                transform_16[2, 0],
                transform_16[2, 1],
                transform_16[2, 2],
            ]
        )
        column_2 = np.array(
            [
                -transform_26[0, 0] * transform_26[1, 3]
                + transform_26[1, 0] * transform_26[0, 3],
                -transform_26[0, 1] * transform_26[1, 3]
                + transform_26[1, 1] * transform_26[0, 3],
                -transform_26[0, 2] * transform_26[1, 3]
                + transform_26[1, 2] * transform_26[0, 3],
                transform_26[2, 0],
                transform_26[2, 1],
                transform_26[2, 2],
            ]
        )
        column_3 = np.array(
            [
                -transform_36[0, 0] * transform_36[1, 3]
                + transform_36[1, 0] * transform_36[0, 3],
                -transform_36[0, 1] * transform_36[1, 3]
                + transform_36[1, 1] * transform_36[0, 3],
                -transform_36[0, 2] * transform_36[1, 3]
                + transform_36[1, 2] * transform_36[0, 3],
                transform_36[2, 0],
                transform_36[2, 1],
                transform_36[2, 2],
            ]
        )
        column_4 = np.array(
            [
                -transform_46[0, 0] * transform_46[1, 3]
                + transform_46[1, 0] * transform_46[0, 3],
                -transform_46[0, 1] * transform_46[1, 3]
                + transform_46[1, 1] * transform_46[0, 3],
                -transform_46[0, 2] * transform_46[1, 3]
                + transform_46[1, 2] * transform_46[0, 3],
                transform_46[2, 0],
                transform_46[2, 1],
                transform_46[2, 2],
            ]
        )
        column_5 = np.array(
            [
                -transform_56[0, 0] * transform_56[1, 3]
                + transform_56[1, 0] * transform_56[0, 3],
                -transform_56[0, 1] * transform_56[1, 3]
                + transform_56[1, 1] * transform_56[0, 3],
                -transform_56[0, 2] * transform_56[1, 3]
                + transform_56[1, 2] * transform_56[0, 3],
                transform_56[2, 0],
                transform_56[2, 1],
                transform_56[2, 2],
            ]
        )
        column_6 = np.array([0, 0, 0, 0, 0, 1])

        rotation = transform_06[:3, :3]
        base_transform = np.zeros((6, 6))
        base_transform[:3, :3] = rotation
        base_transform[3:, 3:] = rotation
        return base_transform @ np.column_stack(
            [column_1, column_2, column_3, column_4, column_5, column_6]
        )

    @staticmethod
    def _pose_error(current_transform, target_transform):
        position_error = (
            current_transform[:3, 3] - target_transform[:3, 3]
        )
        rotation_error = (
            current_transform[:3, :3] @ target_transform[:3, :3].T
        )
        cos_theta = np.clip((np.trace(rotation_error) - 1) / 2.0, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        if np.abs(theta) < 1e-7:
            angular_error = np.zeros(3)
        else:
            axis = np.array(
                [
                    rotation_error[2, 1] - rotation_error[1, 2],
                    rotation_error[0, 2] - rotation_error[2, 0],
                    rotation_error[1, 0] - rotation_error[0, 1],
                ]
            ) / (2 * np.sin(theta))
            angular_error = theta * axis
        return np.hstack((position_error, angular_error))

    def gn_ik(
        self,
        target_pose,
        initial_q,
        max_iter=100,
        tol=1e-7,
        attempts=30,
        record_trace=None,
    ):
        """原 `robot.py` 中供轨迹细分调试调用的高斯-牛顿逆解。"""
        target_transform = self.pose2homography(target_pose)
        if initial_q is not None:
            initial_q = np.array(initial_q, float)

        best_result = None
        for attempt in range(attempts):
            if initial_q is not None and attempt == 0:
                q = initial_q.copy()
            else:
                q = np.random.uniform(self.q_min, self.q_max)

            for iteration in range(max_iter):
                _, current_transform = self.fkine(q)
                error = self._pose_error(current_transform, target_transform)
                if np.linalg.norm(error) < tol:
                    break

                jacobian = self.get_jacobian(q)
                hessian = jacobian.T @ jacobian
                gradient = jacobian.T @ error
                try:
                    delta_q = -np.linalg.solve(hessian, gradient)
                except np.linalg.LinAlgError:
                    delta_q = -np.linalg.pinv(jacobian) @ error
                q = q + 0.01 * delta_q

            _, final_transform = self.fkine(q)
            final_error = np.linalg.norm(
                self._pose_error(final_transform, target_transform)
            )
            result = {
                "q": q.copy(),
                "success": final_error < tol,
                "error": final_error,
                "iterations": iteration if final_error < tol else max_iter,
            }
            if best_result is None or result["error"] < best_result["error"]:
                best_result = result
            if result["success"]:
                return result
        return best_result

    @staticmethod
    def _rotation2rpy(rotation):
        if rotation.shape == (4, 4):
            rotation = rotation[:3, :3]

        pitch = np.arctan2(
            -rotation[2, 0],
            np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2),
        )
        if np.abs(np.cos(pitch)) < 1e-10:
            yaw = 0.0
            roll = np.arctan2(
                np.sign(pitch) * rotation[0, 1],
                rotation[1, 1],
            )
        else:
            roll = np.arctan2(rotation[2, 1], rotation[2, 2])
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.array([roll, pitch, yaw])

    @staticmethod
    def _rpy2rotation(euler):
        rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(euler[0]), -np.sin(euler[0])],
                [0, np.sin(euler[0]), np.cos(euler[0])],
            ]
        )
        ry = np.array(
            [
                [np.cos(euler[1]), 0, np.sin(euler[1])],
                [0, 1, 0],
                [-np.sin(euler[1]), 0, np.cos(euler[1])],
            ]
        )
        rz = np.array(
            [
                [np.cos(euler[2]), -np.sin(euler[2]), 0],
                [np.sin(euler[2]), np.cos(euler[2]), 0],
                [0, 0, 1],
            ]
        )
        return rz @ ry @ rx

    @staticmethod
    def _get_homography_from_R_P(rotation, position):
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.array(position).flatten()
        return transform

    def homography2pose(self, transform):
        roll, pitch, yaw = self._rotation2rpy(transform)
        return np.array(
            [
                transform[0, 3],
                transform[1, 3],
                transform[2, 3],
                roll,
                pitch,
                yaw,
            ]
        )

    def pose2homography(self, pose):
        x, y, z, roll, pitch, yaw = pose
        transform = np.eye(4)
        transform[:3, :3] = self._rpy2rotation([roll, pitch, yaw])
        transform[:3, 3] = [x, y, z]
        return transform

    @staticmethod
    def _rotation_to_axis_angle(rotation):
        angle = np.arccos((np.trace(rotation) - 1) / 2)
        if angle < 1e-6:
            return np.array([1, 0, 0]), 0
        axis = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        ) / (2 * np.sin(angle))
        return axis / np.linalg.norm(axis), angle

    @staticmethod
    def _axis_angle_to_rotation(axis, theta):
        if theta < 1e-6:
            return np.eye(3)
        a, b, c = axis
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array(
            [
                [
                    a * a * (1 - cos_theta) + cos_theta,
                    a * b * (1 - cos_theta) - c * sin_theta,
                    a * c * (1 - cos_theta) + b * sin_theta,
                ],
                [
                    a * b * (1 - cos_theta) + c * sin_theta,
                    b * b * (1 - cos_theta) + cos_theta,
                    b * c * (1 - cos_theta) - a * sin_theta,
                ],
                [
                    a * c * (1 - cos_theta) - b * sin_theta,
                    b * c * (1 - cos_theta) + a * sin_theta,
                    c * c * (1 - cos_theta) + cos_theta,
                ],
            ]
        )

    def linear_interpolation(self, start_pose, target_pose, steps=4):
        start_transform = self.pose2homography(start_pose)
        end_transform = self.pose2homography(target_pose)
        start_rotation = start_transform[:3, :3]
        end_rotation = end_transform[:3, :3]
        start_position = start_transform[:3, 3]
        end_position = np.array(target_pose[:3])
        axis, angle = self._rotation_to_axis_angle(
            start_rotation.T @ end_rotation
        )

        trajectory = []
        for index in range(1, steps + 1):
            progress = index / steps
            position = (
                (1 - progress) * start_position + progress * end_position
            )
            rotation = start_rotation @ self._axis_angle_to_rotation(
                axis,
                angle * progress,
            )
            transform = self._get_homography_from_R_P(rotation, position)
            trajectory.append(self.homography2pose(transform))
        return trajectory


class Pb_Init_:
    """
    PyBullet 场景初始化类。这里专门存放可视化时要重复使用的初始化操作。
    """

    def __init__(self, pb_config=Pb_Config):
        """
        功能：初始化 PyBullet 场景控制类
        输入：
            pb_config: PyBullet 调试配置类
        输出：
            无
        """
        self.pb_config = pb_config

    def cur2pb_pose(self, xyz):
        """
        功能：把毫米单位的位置转换成 PyBullet 需要的米单位位置
        输入：
            xyz: 空间位置 [x,y,z](单位: mm)
        输出：
            pb_xyz: PyBullet 位置列表 [x,y,z](单位: m)
        """
        xyz = np.array(xyz, dtype=float)
        return [float(xyz[0]) / 1000.0, float(xyz[1]) / 1000.0, float(xyz[2]) / 1000.0]

    def init_scene(self, use_gui=True):
        """
        功能：初始化 PyBullet 场景，包括地面、车体、机械臂和末端工具
        输入：
            use_gui: 是否打开图形界面，True 为 GUI，False 为 DIRECT
        输出：
            scene_dict: 场景对象 id 字典，包含 car_id、robot_id、tcp_id
        """
        if p.isConnected():
            p.disconnect()

        os.chdir(self.pb_config.base_dir)

        if use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.loadURDF("plane.urdf")

        p.setAdditionalSearchPath(self.pb_config.pybullet_asset_dir)

        car_id = self.load_car()
        robot_id = self.load_robot()
        tcp_id = self.create_tcp_body()
        self.set_camera()

        scene_dict = {
            "car_id": car_id,
            "robot_id": robot_id,
            "tcp_id": tcp_id,
        }
        return scene_dict

    def load_car(self):
        """
        功能：加载灰色车体模型
        输入：
            无
        输出：
            car_id: 车体在 PyBullet 中的 id
        """
        car_vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=self.pb_config.pybullet_car_stl_path,
            meshScale=[1, 1, 1],
            rgbaColor=self.pb_config.car_color,
        )
        car_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=car_vis,
            basePosition=[0, 0, 0],
        )
        return car_id

    def load_robot(self):
        """
        功能：加载机械臂模型
        输入：
            无
        输出：
            robot_id: 机械臂在 PyBullet 中的 id
        """
        arm_quat = p.getQuaternionFromEuler(self.pb_config.arm_rot)
        robot_id = p.loadURDF(
            self.pb_config.pybullet_robot_urdf_file,
            self.cur2pb_pose(self.pb_config.arm_pos),
            arm_quat,
            useFixedBase=True,
        )
        return robot_id

    def create_tcp_body(self):
        """
        功能：创建末端工具的可视化圆柱体
        输入：
            无
        输出：
            tcp_id: 末端工具在 PyBullet 中的 id
        """
        tool_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.pb_config.tool_radius / 1000.0,
            length=self.pb_config.tool_length / 1000.0,
            rgbaColor=self.pb_config.tool_color,
        )
        tool_col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.pb_config.tool_radius / 1000.0,
            height=self.pb_config.tool_length / 1000.0,
        )
        tcp_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tool_col,
            baseVisualShapeIndex=tool_vis,
            basePosition=[0, 0, 2],
        )
        return tcp_id

    def set_camera(self):
        """
        功能：设置 PyBullet 调试窗口的相机视角
        输入：
            无
        输出：
            无
        """
        p.resetDebugVisualizerCamera(
            cameraDistance=self.pb_config.camera_distance,
            cameraYaw=self.pb_config.camera_yaw,
            cameraPitch=self.pb_config.camera_pitch,
            cameraTargetPosition=self.cur2pb_pose(self.pb_config.camera_target),
        )

    def disconnect(self):
        """
        功能：断开 PyBullet 连接
        输入：
            无
        输出：
            无
        """
        if p.isConnected():
            p.disconnect()


class Pb_Func:
    def __init__(self, pb_config=Pb_Config):
        """
        功能：初始化 PyBullet 调试功能类
        输入：
            pb_config: PyBullet 调试配置类
        输出：
            无
        """
        self.pb_config = pb_config  # 保存配置类，后面所有参数都从这里取
        self.arm_quat = p.getQuaternionFromEuler(self.pb_config.arm_rot)  # 提前把机械臂底座欧拉角转成四元数

    def cur2pb_pose(self, xyz):
        """
        功能：把毫米单位的位置转换成 PyBullet 需要的米单位位置
        输入：
            xyz: 空间位置 [x,y,z](单位: mm)
        输出：
            pb_xyz: PyBullet 位置列表 [x,y,z](单位: m)
        """
        xyz = np.array(xyz, dtype=float)  # 先转成浮点数组，避免后面参与运算时报类型问题
        return [float(xyz[0]) / 1000.0, float(xyz[1]) / 1000.0, float(xyz[2]) / 1000.0]  #  mm 转 m

    def is_key_hit(self, keys, key_char):
        """
        功能：判断某个按键是否被按下
        输入：
            keys: PyBullet 当前键盘事件字典
            key_char: 要检测的按键字符，如 "D"、"A" 或 "Q"
        输出：
            hit_flag: 是否按下该按键，True 或 False
        """
        lower_code = ord(key_char.lower())  # 把目标按键转成小写的 ASCII 编码
        upper_code = ord(key_char.upper())  # 把目标按键转成大写的 ASCII 编码

        if lower_code in keys and keys[lower_code] & p.KEY_WAS_TRIGGERED:  # 小写键被触发
            return True
        if upper_code in keys and keys[upper_code] & p.KEY_WAS_TRIGGERED:  # 大写键被触发
            return True
        return False  # 两种情况都没命中，说明这个键这次没有按下

    def draw_text(self, text_id, text, text_pos, text_color=None, text_size=1.2):
        """
        功能：在 PyBullet 场景中创建或更新一段文字
        输入：
            text_id: 旧文字的 id，没有旧文字时传 None
            text: 要显示的文字内容
            text_pos: 文字位置 [x,y,z](单位: mm)
            text_color: 文字颜色 [r,g,b]
            text_size: 文字大小
        输出：
            text_id: 新建或更新后的文字 id
        """
        if text_color is None:
            text_color = [0, 0, 0]  # 默认用黑色文字

        text_pos = self.cur2pb_pose(text_pos)  # 先把文字位置从 mm 转成 m
        if text_id is None:  # 第一次创建文字
            return p.addUserDebugText(
                text,
                text_pos,
                text_color,
                textSize=text_size,
                lifeTime=0,
            )

        p.addUserDebugText(  # 已经存在的话就原地替换更新
            text,
            text_pos,
            text_color,
            textSize=text_size,
            lifeTime=0,
            replaceItemUniqueId=text_id,
        )
        return text_id

    def draw_line(self, line_id, start_pos, end_pos, color, width):
        """
        功能：在 PyBullet 场景中创建或更新一条线
        输入：
            line_id: 旧线条的 id，没有旧线条时传 None
            start_pos: 起点 [x,y,z](单位: mm)
            end_pos: 终点 [x,y,z](单位: mm)
            color: 线条颜色 [r,g,b]
            width: 线宽
        输出：
            line_id: 新建或更新后的线条 id
        """
        start_pos = self.cur2pb_pose(start_pos)  # 线段起点转成米
        end_pos = self.cur2pb_pose(end_pos)  # 线段终点转成米

        if line_id is None:  # 第一次创建线段
            return p.addUserDebugLine(start_pos, end_pos, color, width, 0)

        p.addUserDebugLine(  # 已有线段就直接替换更新
            start_pos,
            end_pos,
            color,
            width,
            0,
            replaceItemUniqueId=line_id,
        )
        return line_id

    def pose_to_world_point(self, pose):
        """
        功能：把机械臂局部位姿中的位置换算成场景世界坐标中的位置
        输入：
            pose: 末端位姿 [x,y,z,r,p,y](单位: mm,rad)
        输出：
            world_point: 世界坐标位置 [x,y,z](单位: mm)
        """
        pose = np.array(pose, dtype=float)  # 转成浮点数组方便切片和计算
        world_point_pb, _ = p.multiplyTransforms(
            self.cur2pb_pose(self.pb_config.arm_pos),  # 机械臂底座在场景里的位置
            self.arm_quat,  # 机械臂底座在场景里的朝向
            self.cur2pb_pose(pose[:3]),  # 局部坐标里的末端位置
            [0.0, 0.0, 0.0, 1.0],  # 这里只做位置变换，不额外旋转
        )
        return np.array(world_point_pb, dtype=float) * 1000.0  # 转回 mm，方便和主程序统一

    def get_tcp_data(self, solver, joint_values):
        """
        功能：根据关节角计算末端工具在局部坐标和世界坐标下的数据
        输入：
            solver: 机械臂运动学模型
            joint_values: 关节角列表 [j1,...,j6](单位: rad)
        输出：
            tcp_data: 包含 local_pose、world_pose、tcp_world、tcp_center_world、world_quat 的字典
        """
        joint_values = np.array(joint_values, dtype=float)  # 保证关节角是浮点数组
        _, pose_h = solver.fkine(joint_values)  # 用正解算当前关节角对应的位姿矩阵
        local_pose = np.array(solver.homography2pose(pose_h), dtype=float)  # 再把位姿矩阵转成 xyzrpy

        local_xyz = local_pose[:3]  # 末端位置
        local_rpy = local_pose[3:]  # 末端姿态
        local_quat = p.getQuaternionFromEuler(local_rpy.tolist())  # rpy 转四元数

        world_xyz_pb, world_quat = p.multiplyTransforms(
            self.cur2pb_pose(self.pb_config.arm_pos),  # 先给上机械臂底座位置
            self.arm_quat,  # 再给上机械臂底座姿态
            self.cur2pb_pose(local_xyz),  # 乘上末端局部位置
            local_quat,  # 乘上末端局部姿态
        )
        world_xyz = np.array(world_xyz_pb, dtype=float) * 1000.0  # 转回 mm
        world_rpy = np.array(p.getEulerFromQuaternion(world_quat), dtype=float)  # 世界姿态四元数转回 rpy
        world_pose = np.concatenate((world_xyz, world_rpy))  # 拼成世界坐标下的 xyzrpy

        tcp_center_pb, _ = p.multiplyTransforms(
            self.cur2pb_pose(world_xyz),  # 末端法兰位置
            world_quat,  # 末端法兰姿态
            [0.0, 0.0, -self.pb_config.tool_length / 2000.0],  # 圆柱工具中心相对法兰的偏移
            [0.0, 0.0, 0.0, 1.0],  # 工具中心偏移不再附加旋转
        )
        tcp_center_world = np.array(tcp_center_pb, dtype=float) * 1000.0  # 转回 mm

        return {
            "local_pose": local_pose,  # 机械臂自身坐标下的末端位姿
            "world_pose": world_pose,  # 场景世界坐标下的末端位姿
            "tcp_world": world_xyz,  # 末端法兰在世界里的位置
            "tcp_center_world": tcp_center_world,  # 圆柱工具中心在世界里的位置
            "world_quat": world_quat,  # 世界姿态四元数，后面摆工具和画轴要用
        }

    def update_pose_text(self, text_state, local_pose):
        """
        功能：更新界面中的 Base XYZ 和 Base RPY 两行文字
        输入：
            text_state: 保存文字 id 的字典
            local_pose: 当前末端位姿 [x,y,z,r,p,y](单位: mm,rad)
        输出：
            无
        """
        text_anchor = np.array(self.pb_config.arm_pos, dtype=float)  # 先取机械臂底座位置
        text_anchor = text_anchor + np.array(self.pb_config.pose_text_anchor_offset, dtype=float)  # 加上文字偏移
        xyz_pos = text_anchor  # 第一行文字位置
        rpy_pos = text_anchor + np.array([0.0, 0.0, self.pb_config.pose_text_gap], dtype=float)  # 第二行往上错开

        xyz_text = "Base XYZ: " + " ".join(f"{float(v):.3f}" for v in local_pose[:3])  # 组合 xyz 文字
        rpy_text = "Base RPY: " + " ".join(f"{float(v):.3f}" for v in local_pose[3:])  # 组合 rpy 文字

        text_state["xyz"] = self.draw_text(text_state["xyz"], xyz_text, xyz_pos)  # 刷新第一行
        text_state["rpy"] = self.draw_text(text_state["rpy"], rpy_text, rpy_pos)  # 刷新第二行

    def update_axis_lines(self, axis_state, world_pose):
        """
        功能：更新末端工具的 xyz 三个坐标轴显示，并给三根轴加上 x、y、z 标识
        输入：
            axis_state: 保存三根轴线和三个文字 id 的字典
            world_pose: 末端在世界坐标中的位姿 [x,y,z,r,p,y](单位: mm,rad)
        输出：
            无
        """
        tip_world = np.array(world_pose[:3], dtype=float)  # 当前末端位置
        tip_world_pb = self.cur2pb_pose(tip_world)  # 末端位置转成米
        tip_quat = p.getQuaternionFromEuler(world_pose[3:].tolist())  # 当前末端姿态转成四元数

        axis_x_end_pb, _ = p.multiplyTransforms(
            tip_world_pb,  # 坐标轴起点就是末端位置
            tip_quat,  # 跟随末端姿态
            [self.pb_config.axis_length / 1000.0, 0.0, 0.0],  # x 轴正方向延伸
            [0.0, 0.0, 0.0, 1.0],
        )
        axis_y_end_pb, _ = p.multiplyTransforms(
            tip_world_pb,
            tip_quat,
            [0.0, self.pb_config.axis_length / 1000.0, 0.0],  # y 轴正方向延伸
            [0.0, 0.0, 0.0, 1.0],
        )
        axis_z_end_pb, _ = p.multiplyTransforms(
            tip_world_pb,
            tip_quat,
            [0.0, 0.0, self.pb_config.axis_length / 1000.0],  # z 轴正方向延伸
            [0.0, 0.0, 0.0, 1.0],
        )

        axis_x_text_pb, _ = p.multiplyTransforms(
            tip_world_pb,
            tip_quat,
            [self.pb_config.axis_length / 1000.0 + 0.03, 0.0, 0.0],  # x 字母放在 x 轴前面一点
            [0.0, 0.0, 0.0, 1.0],
        )
        axis_y_text_pb, _ = p.multiplyTransforms(
            tip_world_pb,
            tip_quat,
            [0.0, self.pb_config.axis_length / 1000.0 + 0.03, 0.0],  # y 字母放在 y 轴前面一点
            [0.0, 0.0, 0.0, 1.0],
        )
        axis_z_text_pb, _ = p.multiplyTransforms(
            tip_world_pb,
            tip_quat,
            [0.0, 0.0, self.pb_config.axis_length / 1000.0 + 0.03],  # z 字母放在 z 轴前面一点
            [0.0, 0.0, 0.0, 1.0],
        )

        axis_state["x"] = self.draw_line(
            axis_state["x"],  # 保存或替换旧的 x 轴线
            tip_world,
            np.array(axis_x_end_pb, dtype=float) * 1000.0,
            [1, 0, 0],  # x 轴用红色
            2,
        )
        axis_state["y"] = self.draw_line(
            axis_state["y"],  # 保存或替换旧的 y 轴线
            tip_world,
            np.array(axis_y_end_pb, dtype=float) * 1000.0,
            [0, 1, 0],  # y 轴用绿色
            2,
        )
        axis_state["z"] = self.draw_line(
            axis_state["z"],  # 保存或替换旧的 z 轴线
            tip_world,
            np.array(axis_z_end_pb, dtype=float) * 1000.0,
            [0, 0, 1],  # z 轴用蓝色
            2,
        )

        axis_state["x_text"] = self.draw_text(
            axis_state["x_text"],  # 保存或替换旧的 x 字母
            "x",
            np.array(axis_x_text_pb, dtype=float) * 1000.0,
            text_color=[1, 0, 0],
            text_size=1.3,
        )
        axis_state["y_text"] = self.draw_text(
            axis_state["y_text"],  # 保存或替换旧的 y 字母
            "y",
            np.array(axis_y_text_pb, dtype=float) * 1000.0,
            text_color=[0, 0.7, 0],
            text_size=1.3,
        )
        axis_state["z_text"] = self.draw_text(
            axis_state["z_text"],  # 保存或替换旧的 z 字母
            "z",
            np.array(axis_z_text_pb, dtype=float) * 1000.0,
            text_color=[0, 0, 1],
            text_size=1.3,
        )

    def show_robot(self, robot_id, tcp_id, solver, joint_values, text_state=None, axis_state=None):
        """
        功能：把机械臂和末端工具更新到指定关节角，并刷新文字和坐标轴
        输入：
            robot_id: PyBullet 中机械臂的 id
            tcp_id: PyBullet 中末端工具的 id
            solver: 机械臂运动学模型
            joint_values: 目标关节角 [j1,...,j6](单位: rad)
            text_state: 保存文字 id 的字典，默认不刷新文字
            axis_state: 保存坐标轴 id 的字典，默认不刷新坐标轴
        输出：
            tcp_data: 当前末端位姿和工具位置数据字典
        """
        joint_values = np.array(joint_values, dtype=float)  # 保证关节角是浮点数组

        for i in range(len(self.pb_config.joint_indices)):  # 逐个关节写入目标角度
            joint_id = self.pb_config.joint_indices[i]
            p.resetJointState(robot_id, joint_id, float(joint_values[i]))

        tcp_data = self.get_tcp_data(solver, joint_values)  # 计算当前关节角对应的末端数据

        if tcp_id is not None:  # 如果场景里有末端工具模型，就同步它的位置和姿态
            p.resetBasePositionAndOrientation(
                tcp_id,
                self.cur2pb_pose(tcp_data["tcp_center_world"]),
                tcp_data["world_quat"],
            )

        p.performCollisionDetection()  # 这里只刷新碰撞检测，不做物理步进，避免机械臂发软

        if text_state is not None:  # 需要的话刷新界面文字
            self.update_pose_text(text_state, tcp_data["local_pose"])
        if axis_state is not None:  # 需要的话刷新末端 xyz 坐标轴
            self.update_axis_lines(axis_state, tcp_data["world_pose"])

        return tcp_data

    def draw_traj_preview(self, current_pose, traj_poses):
        """
        功能：绘制主插补点组成的绿色直线轨迹和绿色圆点
        输入：
            current_pose: 起始位姿 [x,y,z,r,p,y](单位: mm,rad)
            traj_poses: 主插补位姿列表
        输出：
            world_points: 轨迹上各个主点在世界坐标中的位置列表
        """
        world_points = [self.pose_to_world_point(current_pose)]  # 第一个点是当前起点

        for pose in traj_poses:  # 把后面的主插补点都换算成世界坐标
            world_points.append(self.pose_to_world_point(pose))

        for i in range(1, len(world_points)):  # 逐段画绿色直线
            self.draw_line(
                None,
                world_points[i - 1],
                world_points[i],
                self.pb_config.preview_line_color,
                self.pb_config.preview_line_width,
            )

        point_positions = []  # 保存所有主点的位置
        point_colors = []  # 保存所有主点的颜色
        for point in world_points:
            point_positions.append(self.cur2pb_pose(point))
            point_colors.append(self.pb_config.preview_line_color)

        p.addUserDebugPoints(  # 一次性把所有主点画成明显的绿色圆点
            point_positions,
            point_colors,
            pointSize=self.pb_config.preview_point_size,
            lifeTime=0,
        )
        return world_points

    def clear_progress_lines(self, progress_state):
        """
        功能：清除已经走过轨迹的高亮线
        输入：
            progress_state: 保存高亮线 id 的字典
        输出：
            无
        """
        for item_ids in progress_state.values():
            for item_id in item_ids:
                if item_id is not None:
                    p.removeUserDebugItem(item_id)
            for i in range(len(item_ids)):
                item_ids[i] = None

    def clear_progress_segment(self, progress_state, segment_index):
        """
        功能：清除某一段主插补轨迹的高亮线
        输入：
            progress_state: 保存高亮线 id 的字典
            segment_index: 要清除的轨迹段序号
        输出：
            无
        """
        item_id = progress_state["green"][segment_index]
        if item_id is not None:
            p.removeUserDebugItem(item_id)
            progress_state["green"][segment_index] = None

    def fill_progress_segment(self, world_points, progress_state, segment_index):
        """
        功能：把某一段主插补轨迹高亮到整段结束
        输入：
            world_points: 主轨迹各点的世界坐标位置列表
            progress_state: 保存高亮线 id 的字典
            segment_index: 要填满的轨迹段序号
        输出：
            无
        """
        self.update_progress_line(
            world_points,
            progress_state,
            segment_index,
            world_points[segment_index + 1],
        )

    def sync_progress_to_index(self, world_points, progress_state, current_index):
        """
        功能：根据当前所在主插补点刷新整条已走轨迹的高亮状态
        输入：
            world_points: 主轨迹各点的世界坐标位置列表
            progress_state: 保存高亮线 id 的字典
            current_index: 当前主插补点序号，-1 表示初始点
        输出：
            无
        """
        self.clear_progress_lines(progress_state)
        for segment_index in range(current_index + 1):
            self.fill_progress_segment(world_points, progress_state, segment_index)

    def update_progress_line(self, world_points, progress_state, segment_index, end_world):
        """
        功能：更新当前已经走过的那一段轨迹高亮
        输入：
            world_points: 主轨迹各个点的世界坐标位置列表
            progress_state: 保存高亮线 id 的字典
            segment_index: 当前是第几段轨迹
            end_world: 当前段已经走到的位置 [x,y,z](单位: mm)
        输出：
            无
        """
        start_world = np.array(world_points[segment_index], dtype=float)
        end_world = np.array(end_world, dtype=float)

        progress_state["green"][segment_index] = self.draw_line(
            progress_state["green"][segment_index],
            start_world,
            end_world,
            self.pb_config.progress_inner_color,
            self.pb_config.progress_inner_width,
        )

    def wait_debug_command(self, auto_play=False):
        """
        功能：等待调试按键输入
        输入：
            auto_play: 是否自动播放，True 时不等待键盘直接继续
        输出：
            command: 返回 "next"、"prev" 或 "quit"
        """
        if auto_play:
            time.sleep(self.pb_config.auto_wait)
            return "next"

        while True:
            keys = p.getKeyboardEvents()
            if self.is_key_hit(keys, "d"):
                return "next"
            if self.is_key_hit(keys, "a"):
                return "prev"
            if self.is_key_hit(keys, "q"):
                return "quit"
            time.sleep(1.0 / 60.0)

    def print_current_step(self, step_index, local_pose, joint_values):
        """
        功能：在终端打印当前主插补点的位姿和关节角
        输入：
            step_index: 当前主插补点序号
            local_pose: 当前位姿 [x,y,z,r,p,y](单位: mm,rad)
            joint_values: 当前关节角 [j1,...,j6](单位: rad)
        输出：
            无
        """
        pose_text = np.round(np.array(local_pose, dtype=float), 3).tolist()  # 位姿保留 3 位小数
        joint_text = np.round(np.array(joint_values, dtype=float), 6).tolist()  # 关节角保留 6 位小数
        print(f"当前插补点 {step_index}")
        # print(f"pose(xyzrpy): {pose_text}")
        # print(f"joint(rad): {joint_text}")

    def build_debug_segment_joints(self, solver, start_pose, end_pose, start_joints, end_joints):
        """
        功能：在两个主插补点之间再细分调试插补点，并逐点做逆解
        输入：
            solver: 机械臂运动学模型
            start_pose: 当前主点位姿 [x,y,z,r,p,y](单位: mm,rad)
            end_pose: 下一个主点位姿 [x,y,z,r,p,y](单位: mm,rad)
            start_joints: 当前主点关节角 [j1,...,j6](单位: rad)
            end_joints: 下一个主点关节角 [j1,...,j6](单位: rad)
        输出：
            fine_joints: 细分调试点对应的关节角列表，失败时返回 None
        """
        fine_poses = solver.linear_interpolation(start_pose, end_pose, steps=self.pb_config.debug_interp_steps)  # 先做细分直线插补
        fine_joints = []  # 保存细分点的逆解结果
        last_joints = np.array(start_joints, dtype=float)  # 第一个细分点的逆解初值用段起点关节角

        for pose in fine_poses:  # 逐个细分点做逆解
            ik_result = solver.gn_ik(pose, initial_q=last_joints)
            if not ik_result["success"]:
                print("调试细分插补点逆解失败，直接跳到当前主插补点。")
                return None

            current_joints = np.array(ik_result["q"], dtype=float)  # 当前细分点逆解出的关节角
            fine_joints.append(current_joints)
            last_joints = current_joints  # 下一次逆解继续用这次结果当初值

        if len(fine_joints) > 0:
            fine_joints[-1] = np.array(end_joints, dtype=float)  # 最后一个点强制对齐主插补点关节角

        return fine_joints

    def animate_one_segment(
        self,
        solver,
        robot_id,
        tcp_id,
        world_points,
        progress_state,
        segment_index,
        start_pose,
        end_pose,
        start_joints,
        end_joints,
        text_state,
        axis_state,
        direction=1,
        auto_play=False,
    ):
        """
        功能：播放一段主轨迹之间的调试动画
        输入：
            solver: 机械臂运动学模型
            robot_id: PyBullet 中机械臂的 id
            tcp_id: PyBullet 中末端工具的 id
            world_points: 主轨迹各个点的世界坐标位置列表
            progress_state: 保存高亮线 id 的字典
            segment_index: 当前播放的是第几段主轨迹
            start_pose: 当前主点位姿 [x,y,z,r,p,y](单位: mm,rad)
            end_pose: 下一个主点位姿 [x,y,z,r,p,y](单位: mm,rad)
            start_joints: 当前主点关节角 [j1,...,j6](单位: rad)
            end_joints: 下一个主点关节角 [j1,...,j6](单位: rad)
            text_state: 保存文字 id 的字典
            axis_state: 保存坐标轴 id 的字典
            auto_play: 是否自动播放
        输出：
            result: 播放结果，"end"、"start" 或 "quit"
            tcp_data: 当前末端位姿和工具位置数据字典
        """
        fine_joints = self.build_debug_segment_joints(
            solver,
            start_pose,
            end_pose,
            start_joints,
            end_joints,
        )  # 先准备好这一段所有细分点关节角

        if fine_joints is None:  # 细分逆解失败，就直接跳到主终点
            tcp_data = self.show_robot(robot_id, tcp_id, solver, end_joints, text_state, axis_state)
            if direction > 0:
                self.fill_progress_segment(world_points, progress_state, segment_index)
            else:
                self.clear_progress_segment(progress_state, segment_index)
            return "end", tcp_data

        p.getKeyboardEvents()  # 先清一下上一轮残留按键
        ignore_same_direction_count = 2  # 防止刚进入时把启动这段的按键当成这段的跳过

        for current_joints in fine_joints:  # 逐个细分点播放
            keys = p.getKeyboardEvents()

            if self.is_key_hit(keys, "q"):  # 中途按 Q 直接退出
                tcp_data = self.show_robot(robot_id, tcp_id, solver, current_joints, text_state, axis_state)
                self.update_progress_line(world_points, progress_state, segment_index, tcp_data["tcp_world"])
                return "quit", tcp_data

            if not auto_play:
                if direction > 0:
                    if ignore_same_direction_count <= 0 and self.is_key_hit(keys, "d"):  # 中途按 D 直接跳到这一段终点
                        tcp_data = self.show_robot(robot_id, tcp_id, solver, end_joints, text_state, axis_state)
                        self.fill_progress_segment(world_points, progress_state, segment_index)
                        return "end", tcp_data
                    if self.is_key_hit(keys, "a"):  # 中途按 A 直接退回这一段起点
                        tcp_data = self.show_robot(robot_id, tcp_id, solver, start_joints, text_state, axis_state)
                        self.clear_progress_segment(progress_state, segment_index)
                        return "start", tcp_data
                else:
                    if ignore_same_direction_count <= 0 and self.is_key_hit(keys, "a"):  # 中途按 A 直接跳到上一点
                        tcp_data = self.show_robot(robot_id, tcp_id, solver, end_joints, text_state, axis_state)
                        self.clear_progress_segment(progress_state, segment_index)
                        return "end", tcp_data
                    if self.is_key_hit(keys, "d"):  # 反向播放中按 D 回到这一段起点
                        tcp_data = self.show_robot(robot_id, tcp_id, solver, start_joints, text_state, axis_state)
                        self.fill_progress_segment(world_points, progress_state, segment_index)
                        return "start", tcp_data

            tcp_data = self.show_robot(robot_id, tcp_id, solver, current_joints, text_state, axis_state)  # 正常播放当前细分点
            self.update_progress_line(world_points, progress_state, segment_index, tcp_data["tcp_world"])  # 同时推进已走高亮轨迹

            if ignore_same_direction_count > 0:
                ignore_same_direction_count = ignore_same_direction_count - 1  # 前几帧忽略掉连按误触

            time.sleep(self.pb_config.debug_interp_dt)  # 控制这一段的播放速度

        tcp_data = self.show_robot(robot_id, tcp_id, solver, end_joints, text_state, axis_state)  # 最终强制对齐主终点
        if direction > 0:
            self.fill_progress_segment(world_points, progress_state, segment_index)
        else:
            self.clear_progress_segment(progress_state, segment_index)
        return "end", tcp_data

    def visualize_traj_by_key(
        self,
        solver,
        robot_id,
        tcp_id,
        start_joints,
        traj_joints,
        current_pose,
        traj_poses,
        auto_play=False,
        failure_info=None,
    ):
        """
        功能：在 PyBullet 中按键显示整条主轨迹的调试过程
        输入：
            solver: 机械臂运动学模型
            robot_id: PyBullet 中机械臂的 id
            tcp_id: PyBullet 中末端工具的 id
            start_joints: 初始关节角 [j1,...,j6](单位: rad)
            traj_joints: 主插补点对应的关节角列表
            current_pose: 初始位姿 [x,y,z,r,p,y](单位: mm,rad)
            traj_poses: 主插补位姿列表
            auto_play: 是否自动播放
            failure_info: 主插补逆解失败时的错误信息，默认无
        输出：
            show_ok: 是否正常结束可视化，True 或 False
        """
        if traj_poses is None or len(traj_poses) == 0:
            print("没有可以显示的轨迹。")
            return False

        if traj_joints is None:
            traj_joints = []  # 防止后面遍历时报错

        start_joints = np.array(start_joints, dtype=float)  # 初始关节角转成浮点数组
        current_pose = np.array(current_pose, dtype=float)  # 初始位姿转成浮点数组

        main_joint_list = []  # 保存每个主插补点的关节角
        for joint_values in traj_joints:
            main_joint_list.append(np.array(joint_values, dtype=float))

        main_pose_list = []  # 保存每个主插补点的位姿
        for pose in traj_poses:
            main_pose_list.append(np.array(pose, dtype=float))

        text_state = {"xyz": None, "rpy": None}  # 保存两行文字的 id
        axis_state = {
            "x": None,
            "y": None,
            "z": None,
            "x_text": None,
            "y_text": None,
            "z_text": None,
        }  # 保存三根轴线和三个轴字母的 id

        tcp_data = self.show_robot(robot_id, tcp_id, solver, start_joints, text_state, axis_state)  # 先摆到真实初始位置
        world_points = self.draw_traj_preview(current_pose, main_pose_list)  # 画出整条绿色主轨迹
        progress_state = {
            "green": [None] * (len(world_points) - 1),
        }  # 记录每一段已走轨迹的绿线 id

        print("PyBullet 调试可视化已打开。")
        print("按 D 播放到下一个主插补点，按 A 回到上一个主插补点，按 Q 退出。")
        print(f"每两个主插补点之间会再细分 {self.pb_config.debug_interp_steps} 个调试插补点。")
        print("走到最后一个主插补点后，再按 D 会回到初始位置；在初始位置按 A 会瞬移到最后一个主插补点。")

        if failure_info is not None:
            print("当前是逆解失败调试界面。")
            print(f"失败点序号: {failure_info['failed_step']}/{failure_info['total_steps']}")

        if len(main_joint_list) == 0:  # 一个主点都没求出来时，只能给你看失败界面
            print("没有成功求出的主插补点，按 Q 退出。")
            while True:
                command = self.wait_debug_command(auto_play=False)
                if command == "quit":
                    break
            return True

        current_index = -1  # -1 代表现在还在初始位置，还没走到任何主点
        auto_count = 0  # 自动播放时用于计数

        while True:
            if auto_play and auto_count >= len(main_joint_list):  # 自动播放已经走完整条轨迹
                break

            command = self.wait_debug_command(auto_play=auto_play)  # 等待下一条命令
            if command == "quit":
                break

            if command == "next":
                if current_index == len(main_joint_list) - 1:  # 当前已经在最后一个主插补点
                    if auto_play:
                        break

                    current_index = -1  # 重置回初始状态
                    self.clear_progress_lines(progress_state)
                    tcp_data = self.show_robot(robot_id, tcp_id, solver, start_joints, text_state, axis_state)  # 瞬移回初始关节角
                    # print("已回到初始位置。")
                    # print(f"pose(xyzrpy): {np.round(np.array(tcp_data['local_pose']), 3).tolist()}")
                    # print(f"joint(rad): {np.round(np.array(start_joints), 6).tolist()}")
                    continue

                direction = 1
                target_index = current_index + 1  # 这次要去的下一个主点编号
                segment_index = target_index
                end_pose = main_pose_list[target_index]  # 这次的目标主点位姿
                end_joints = main_joint_list[target_index]  # 这次的目标主点关节角

                if current_index < 0:  # 如果当前还在初始位置
                    start_pose = current_pose
                    start_joint_values = start_joints
                else:  # 如果当前已经在某个主点上
                    start_pose = main_pose_list[current_index]
                    start_joint_values = main_joint_list[current_index]
            else:
                if current_index < 0:  # 初始点按 A 直接跳到最后一个主插补点
                    current_index = len(main_joint_list) - 1
                    self.sync_progress_to_index(world_points, progress_state, current_index)
                    tcp_data = self.show_robot(
                        robot_id,
                        tcp_id,
                        solver,
                        main_joint_list[current_index],
                        text_state,
                        axis_state,
                    )
                    print("已瞬移到最后一个主插补点。")
                    self.print_current_step(current_index + 1, tcp_data["local_pose"], main_joint_list[current_index])
                    continue

                direction = -1
                target_index = current_index - 1  # 这次要回到的上一个主点编号
                segment_index = current_index
                start_pose = main_pose_list[current_index]
                start_joint_values = main_joint_list[current_index]

                if target_index < 0:  # 第一个主点再往回就是初始点
                    end_pose = current_pose
                    end_joints = start_joints
                else:
                    end_pose = main_pose_list[target_index]
                    end_joints = main_joint_list[target_index]

            result, tcp_data = self.animate_one_segment(
                solver,
                robot_id,
                tcp_id,
                world_points,
                progress_state,
                segment_index,
                start_pose,
                end_pose,
                start_joint_values,
                end_joints,
                text_state,
                axis_state,
                direction=direction,
                auto_play=auto_play,
            )  # 播放这一段主轨迹

            if result == "quit":
                break

            if result == "end":
                current_index = target_index
                final_joints = end_joints
                if command == "next":
                    auto_count = auto_count + 1  # 自动播放计数加一
            else:
                final_joints = start_joint_values

            if current_index < 0:
                print("已回到初始位置。")
                print(f"pose(xyzrpy): {np.round(np.array(tcp_data['local_pose']), 3).tolist()}")
                print(f"joint(rad): {np.round(np.array(start_joints), 6).tolist()}")
            else:
                self.print_current_step(current_index + 1, tcp_data["local_pose"], final_joints)  # 打印当前主点信息

        return True


def build_demo_trajectory(solver):
    """用已有正运动学从三组关节角构造一条可视化示范轨迹。"""
    joint_points = [
        np.radians([0, 70, -90, 0, -85, 0]),
        np.radians([12, 64, -86, 8, -80, 10]),
        np.radians([-10, 58, -82, -8, -75, -10]),
    ]
    poses = [
        solver.homography2pose(solver.fkine(joints)[1])
        for joints in joint_points
    ]
    return joint_points[0], joint_points[1:], poses[0], poses[1:]


def main(use_gui=True, auto_play=False):
    required_assets = [
        Path(Pb_Config.pybullet_car_stl_path),
        Path(Pb_Config.pybullet_robot_urdf_file),
        PYBULLET_ASSET_DIR / "RML-63" / "meshes",
    ]
    missing_assets = [str(path) for path in required_assets if not path.exists()]
    if missing_assets:
        raise FileNotFoundError("缺少 PyBullet 模型资源:\n" + "\n".join(missing_assets))

    original_working_dir = Path.cwd()
    initializer = Pb_Init_()
    try:
        scene = initializer.init_scene(use_gui=use_gui)
        solver = RobotKinematics(tool_length=Pb_Config.tool_length)
        start_joints, traj_joints, current_pose, traj_poses = (
            build_demo_trajectory(solver)
        )
        Pb_Func().visualize_traj_by_key(
            solver,
            scene["robot_id"],
            scene["tcp_id"],
            start_joints,
            traj_joints,
            current_pose,
            traj_poses,
            auto_play=auto_play,
        )
    finally:
        initializer.disconnect()
        os.chdir(original_working_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auto",
        action="store_true",
        help="自动播放，不等待 D/A 按键。",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="使用无窗口 DIRECT 模式；同时自动播放。",
    )
    arguments = parser.parse_args()
    main(
        use_gui=not arguments.direct,
        auto_play=arguments.auto or arguments.direct,
    )
