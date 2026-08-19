"""외부 ROS 2 색상 인식 결과로 큐브를 분류해 놓는 예제.

실행: isaac_python 7_pick_place_color.py

전체 흐름
  1. 로봇 앞 작업 공간에 파랑 또는 초록 큐브 하나와 색상 마커 두 개를 무작위 생성한다.
  2. 로봇 손목의 RealSense 카메라 영상은 /rgb(sensor_msgs/Image)로 계속 전송된다.
  3. 다른 PC의 색상 인식 노드는 /color_id(std_msgs/Int32)를 보낸다.
       1 = 파랑 마커, 2 = 초록 마커
  4. 이 스크립트는 결과를 기다렸다가 큐브를 집고 해당 색상 마커 위에 내려놓는다.

색상 인식 노드 자체는 이 파일의 범위가 아니다. ROS 2 네트워크에서 /rgb와
/color_id 토픽이 서로 보이도록 두 PC의 ROS_DOMAIN_ID 등을 맞춰야 한다.
"""
# SimulationApp은 Isaac Sim을 독립 파이썬 프로그램으로 실행하는 진입점이다.
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# ROS 2 Bridge 확장을 먼저 켜야 ROS2CameraHelper와 ROS 메시지를 사용할 수 있다.
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

from pathlib import Path
import os
import time
import numpy as np
import omni.usd
import omni.graph.core as og
from pxr import Usd, UsdGeom, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver

# ────────────────────────────────────────────────────────────────────────────
# 파일 경로와 USD(로봇/카메라가 포함된 씬) 설정
# ────────────────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
M0609_DIR = THIS_DIR.parent
USD_PATH = str(M0609_DIR / "Collected_m0609_camera_cube/m0609_camera_cube.usd")
URDF_PATH = str(M0609_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
DESCRIPTION_PATH = str(M0609_DIR / "descriptor/m0609_description.yaml")
# Prim path는 USD 씬 그래프 안의 객체 주소다. 카메라 경로는 요청에서 지정한 값이다.
ROBOT_PRIM_PATH = "/World/m0609"
# USD 내부 /World/Graph/camera_graph가 아래 실제 Color Camera prim으로
# 이미 /rgb를 발행한다. 이 코드에서는 같은 토픽 publisher를 중복 생성하지 않는다.
CAMERA_PRIM_PATH = (
    "/World/m0609/onrobot_rg2ft/angle_bracket/realsense_d455/RSD455/"
    "Camera_OmniVision_OV9782_Color"
)
EE_LINK_NAME = "link_6"
ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
DRIVE_STIFFNESS, DRIVE_DAMPING, DRIVE_MAX_FORCE = 1e8, 1e4, 1e8
GRIPPER_JOINTS = ["finger_joint", "right_inner_knuckle_joint"]
GRIPPER_OPEN_POS, GRIPPER_CLOSE_POS = 0.0, 0.8
READY_JOINTS_DEG = [0, 0, 90, 0, 90, 0]
TCP_OFFSET = np.array([0.0, 0.0, 0.19671])

# ────────────────────────────────────────────────────────────────────────────
# 작업 물체/안전 범위 설정 (Isaac Sim의 길이 단위는 여기서 metre)
# ────────────────────────────────────────────────────────────────────────────
# 원래 10 cm였던 큐브의 한 변을 1/4로 줄였다: 2.5 cm.
# PICK_Z/PLACE_Z는 아래에서 이 값으로부터 자동 계산되므로 함께 맞춰진다.
CUBE_SIZE = 0.05
MARKER_SIZE = np.array([0.16, 0.16, 0.012])
# 이론 최대 도달 반경은 약 0.9 m지만, 베이스 바로 옆과 경계는 피한다.
MIN_RADIUS, MAX_RADIUS = 0.28, 0.5
MIN_SEPARATION = 0.25
# 외부 PC 응답이 늦을 때 상태를 다시 출력하는 주기다. 이 시간이 지나도
# WAIT 상태를 끝내지 않으므로, 나중에 도착한 유효 메시지로 동작할 수 있다.
COLOR_TIMEOUT_SEC = 15.0
# 큐브를 들어 올린 직후에는 PC B에 이전 카메라 프레임의 분류 결과가 남아
# 있을 수 있다. 이 시간 동안 수신 결과를 버리고 새 영상 처리 시간을 준다.
COLOR_SETTLE_SEC = 2.0
# /color_id가 아직 없을 때 같은 안내를 매 simulation frame마다 출력하지 않는다.
INVALID_COLOR_LOG_INTERVAL_SEC = 5.0
# 큐브와 마커 모두 프로젝트에서 사용하던 원래 파랑/초록 값을 사용한다.
BLUE = np.array([0.05, 0.20, 1.0])
GREEN = np.array([0.05, 0.75, 0.15])
COLOR_NAMES = {1: "blue", 2: "green"}
PICK_Z, PLACE_Z, APPROACH_HEIGHT, LIFT_HEIGHT = CUBE_SIZE / 2, CUBE_SIZE / 2 + 0.005, 0.25, 0.23
TCP_SPEED, MIN_STEPS, MAX_STEPS, GRIPPER_WAIT = 0.004, 60, 600, 120
RETURN_HOME_STEPS = 180
# READY 자세에 도착한 뒤 물리가 안정될 시간을 조금 준 후 다음 사이클을 시작한다.
AUTO_RESTART_WAIT_STEPS = 60

# ────────────────────────────────────────────────────────────────────────────
# 자세/좌표 유틸리티
# Lula IK는 플랜지(link_6) 위치를 원하지만, 우리는 손가락 끝(TCP)을 기준으로
# 경로를 만들므로 둘 사이의 TCP_OFFSET을 변환한다.
# ────────────────────────────────────────────────────────────────────────────
def qmul(a, b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
def qaxis(axis, deg):
    axis=np.asarray(axis,float); axis/=np.linalg.norm(axis); h=np.radians(deg)/2
    return np.r_[np.cos(h), axis*np.sin(h)]
def target_quat():
    """그리퍼가 아래를 향하도록 하는 목표 회전(쿼터니언)을 만든다."""
    q=qmul(qaxis([1,0,0],180), qaxis([0,1,0],0)); return q/np.linalg.norm(q)
def qmat(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)], [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)], [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def flange(tcp, q): return np.asarray(tcp)-qmat(q)@TCP_OFFSET
def tcp_pose(robot):
    p,q=robot.end_effector.get_world_pose(); return p+qmat(q)@TCP_OFFSET
def find_prim(root, name):
    root=omni.usd.get_context().get_stage().GetPrimAtPath(root)
    return next((str(p.GetPath()) for p in Usd.PrimRange(root) if p.GetName()==name), None)
def sample_positions():
    """큐브, 파랑 마커, 초록 마커의 XY 위치를 안전하게 무작위로 뽑는다.

    완전한 원형 범위에는 로봇 뒤쪽도 포함되므로, 실제로 안정적으로 도달 가능한
    전방 -70~+70도 부채꼴만 사용한다. 각 물체는 서로 MIN_SEPARATION 이상 떨어진다.
    """
    points=[]
    while len(points)<3:
        r=np.random.uniform(MIN_RADIUS, MAX_RADIUS); a=np.random.uniform(np.radians(-70),np.radians(70))
        p=np.array([r*np.cos(a), r*np.sin(a)])
        if all(np.linalg.norm(p-other)>=MIN_SEPARATION for other in points): points.append(p)
    return points

# ────────────────────────────────────────────────────────────────────────────
# ROS 2 수신부
# rclpy를 쓰지 않고 Isaac Sim ROS 2 Bridge의 OmniGraph만 사용한다.
# ────────────────────────────────────────────────────────────────────────────
class ColorIdReceiver:
    """Bridge 그래프의 std_msgs/Int32 출력과 메시지 수신 횟수를 읽는다."""
    def __init__(self, graph_path):
        self.data = og.Controller.attribute(f"{graph_path}/ColorId.outputs:data")
        self.count = og.Controller.attribute(f"{graph_path}/ColorEventCounter.outputs:count")
        self.start_count = int(self.count.get() or 0)
        self.handled_count = self.start_count
        self.last_invalid_log_time = float("-inf")
        self.ignore_until = 0.0
    def begin_cycle(self):
        """이전 Play/카메라 프레임의 결과를 잠시 버리는 새 검사 주기를 시작한다."""
        self.start_count = int(self.count.get() or 0)
        self.handled_count = self.start_count
        self.ignore_until = time.monotonic() + COLOR_SETTLE_SEC
    def result(self):
        """이번 Play 후 새 메시지가 도착했을 때만 유효한 색상 ID를 반환한다."""
        count = int(self.count.get() or 0)
        # settle 구간에는 counter 기준점도 계속 최신으로 옮겨서 큐에 남은
        # 과거 Int32가 settle 종료 직후 채택되지 않게 한다.
        if time.monotonic() < self.ignore_until:
            self.start_count = count
            self.handled_count = count
            return None
        if count <= self.start_count or count == self.handled_count:
            return None
        self.handled_count = count
        value = int(self.data.get() or 0)
        if value in COLOR_NAMES:
            print(f"received /color_id={value} ({COLOR_NAMES[value]})")
            return value
        # ROS2Subscriber는 새 ROS 메시지가 없을 때에도 기본값 0을 내보낼 수 있다.
        # 콘솔이 0 로그로 채워지지 않도록 안내는 5초마다 한 번만 출력한다.
        now = time.monotonic()
        if now - self.last_invalid_log_time >= INVALID_COLOR_LOG_INTERVAL_SEC:
            print(f"waiting for /color_id: last value {value}; expected 1 or 2")
            self.last_invalid_log_time = now
        return None

# ────────────────────────────────────────────────────────────────────────────
# Pick & Place 상태 기계
# 큐브를 먼저 집어 카메라 앞에 든 후 색상 결과를 기다린다. 분류가 끝나면
# 해당 마커에 놓고 마지막으로 READY_JOINTS_DEG 자세로 복귀한다.
# ────────────────────────────────────────────────────────────────────────────
class PickPlaceFSM:
    APPROACH, DESCEND, GRASP, LIFT, WAIT, MOVE, LOWER, RELEASE, RETURN, DONE = range(10)
    names = ["APPROACH", "DESCEND", "GRASP", "LIFT", "WAIT_COLOR",
             "MOVE", "LOWER", "RELEASE", "RETURN_HOME", "DONE"]

    def __init__(self, robot):
        self.robot = robot
        self.reset()

    def reset(self):
        self.state = self.DONE
        self.step = 0
        self.start = None
        self.goal = None
        self.n = 0
        self.gripper = "open"
        self.goals = {}
        self.wait_started = None
        self.wait_initialized = False
        self.return_start = None

    def begin_pick(self, pick_xy):
        """색상 검사 전에 큐브 접근→하강→파지→들어올리기 경로를 시작한다."""
        self.goals = {
            self.APPROACH: np.r_[pick_xy, APPROACH_HEIGHT],
            self.DESCEND: np.r_[pick_xy, PICK_Z],
            self.LIFT: np.r_[pick_xy, LIFT_HEIGHT],
        }
        self.state = self.APPROACH
        self.step = 0
        self.start = None
        self.goal = None
        self.gripper = "open"
        self.wait_started = None
        self.wait_initialized = False
        self.return_start = None
        print(f"pick sequence started: cube {pick_xy}")

    def set_destination(self, place_xy):
        """색상 ID가 확정된 뒤 마커 이동→하강→놓기 경로를 추가한다."""
        self.goals[self.MOVE] = np.r_[place_xy, LIFT_HEIGHT]
        self.goals[self.LOWER] = np.r_[place_xy, PLACE_Z]
        self.state = self.MOVE
        self.step = 0
        self.start = None
        self.goal = None
        print(f"colour accepted: place {place_xy}")

    def current(self):
        """현재 이동 구간에서 선형 보간한 손가락 끝(TCP) 목표 위치를 반환한다."""
        if self.goal is None or self.start is None:
            return tcp_pose(self.robot)
        return self.start + min(1.0, self.step / float(self.n)) * (self.goal - self.start)

    def advance(self):
        """구간을 시작하거나 한 step 전진한다. GRASP/RELEASE에서는 제자리에서 대기한다."""
        if self.state in (self.WAIT, self.RETURN, self.DONE):
            return
        if self.start is None:
            self.start = tcp_pose(self.robot)
            self.goal = self.start if self.state in (self.GRASP, self.RELEASE) else self.goals[self.state]
            self.gripper = {self.GRASP: "close", self.RELEASE: "open"}.get(self.state, self.gripper)
            self.n = (GRIPPER_WAIT if self.state in (self.GRASP, self.RELEASE)
                      else int(np.clip(np.linalg.norm(self.goal-self.start)/TCP_SPEED,
                                       MIN_STEPS, MAX_STEPS)))
            print(f"{self.names[self.state]} -> {self.goal}, {self.n} steps")
        self.step += 1
        if self.step >= self.n:
            self.state += 1
            self.step = 0
            self.start = None
            self.goal = None

    def return_arm_positions(self):
        """현재 관절각에서 READY_JOINTS_DEG까지 부드럽게 보간한다."""
        if self.return_start is None:
            self.return_start = self.robot.get_joint_positions()[:6].copy()
            self.step = 0
            print(f"RETURN_HOME -> {READY_JOINTS_DEG}, {RETURN_HOME_STEPS} steps")
        target = np.deg2rad(np.asarray(READY_JOINTS_DEG, dtype=float))
        alpha = min(1.0, (self.step + 1) / float(RETURN_HOME_STEPS))
        positions = self.return_start + alpha * (target - self.return_start)
        self.step += 1
        if self.step >= RETURN_HOME_STEPS:
            self.state = self.DONE
            print("DONE: robot returned to the default pose")
        return positions

# ────────────────────────────────────────────────────────────────────────────
# Isaac Sim 씬 구성
# BaseTask.set_up_scene()은 world.reset() 시 한 번 호출되어 USD와 물체를 준비한다.
# ────────────────────────────────────────────────────────────────────────────
class Task(BaseTask):
    def __init__(self): super().__init__("m0609_color_task",offset=None); self.robot=None
    def set_up_scene(self,scene):
        """기존 M0609 USD를 로드하고 로봇, 큐브, 두 마커를 Isaac Scene에 등록한다."""
        super().set_up_scene(scene); stage=omni.usd.get_context().get_stage(); UsdGeom.Xform.Define(stage,"/World"); stage.GetPrimAtPath("/World").GetReferences().AddReference(USD_PATH)
        for _ in range(15): simulation_app.update()
        for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM_PATH)):
            if p.GetName() in ARM_JOINTS:
                for typ in ("angular","linear"):
                    d=UsdPhysics.DriveAPI.Get(p,typ)
                    if d: d.GetStiffnessAttr().Set(DRIVE_STIFFNESS); d.GetDampingAttr().Set(DRIVE_DAMPING); d.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
        ee=find_prim(ROBOT_PRIM_PATH,EE_LINK_NAME)
        # ParallelGripper도 API 인자 순서에 의존하지 않도록 기존 예제와 같은
        # 키워드 방식으로 생성한다.
        g=ParallelGripper(
            end_effector_prim_path=ee,
            joint_prim_names=GRIPPER_JOINTS,
            joint_opened_positions=np.array([GRIPPER_OPEN_POS] * 2),
            joint_closed_positions=np.array([GRIPPER_CLOSE_POS] * 2),
            action_deltas=None,
        )
        # SingleManipulator의 두 번째 위치 인자는 pose의 position이다. 따라서
        # prim/name/EE/gripper는 반드시 키워드로 전달해 인자 순서 혼동을 막는다.
        self.robot=scene.add(SingleManipulator(
            prim_path=ROBOT_PRIM_PATH,
            name="m0609_robot",
            end_effector_prim_path=ee,
            gripper=g,
        ))
        self.cube=scene.add(DynamicCuboid(
            prim_path="/World/colour_cube", name="colour_cube",
            position=np.array([.4, 0, CUBE_SIZE / 2]),
            scale=np.ones(3) * CUBE_SIZE, color=BLUE,
        ))
        self.blue_marker=scene.add(VisualCuboid(
            prim_path="/World/blue_marker", name="blue_marker",
            position=np.array([.5, -.2, MARKER_SIZE[2] / 2]),
            scale=MARKER_SIZE, color=BLUE,
        ))
        self.green_marker=scene.add(VisualCuboid(
            prim_path="/World/green_marker", name="green_marker",
            position=np.array([.5, .2, MARKER_SIZE[2] / 2]),
            scale=MARKER_SIZE, color=GREEN,
        ))
    def randomize(self):
        """새 사이클마다 물체 위치와 큐브 색을 새로 정한다.

        실제 큐브 색은 이 프로그램이 알지만, 동작 목표는 반드시 외부에서 수신한
        /color_id로 정한다. 따라서 카메라-ROS-외부 인식 노드 연결을 검증할 수 있다.
        """
        cube,blue,green=sample_positions(); colour=int(np.random.choice([1,2]))
        self.cube.set_world_pose(np.r_[cube,CUBE_SIZE/2])
        # 이전 place 동작에서 남은 속도를 제거해야 새 위치에서 큐브가 튀지 않는다.
        self.cube.set_linear_velocity(np.zeros(3))
        self.cube.set_angular_velocity(np.zeros(3))
        # DynamicCuboid에는 set_color()가 없다. 생성 때 적용된 PreviewSurface
        # material을 통해서만 실제 렌더 색상을 변경할 수 있다.
        self.cube.get_applied_visual_material().set_color(
            BLUE if colour == 1 else GREEN
        )
        self.blue_marker.set_world_pose(np.r_[blue,MARKER_SIZE[2]/2]); self.green_marker.set_world_pose(np.r_[green,MARKER_SIZE[2]/2]); self.pick_xy=cube; self.destinations={1:blue,2:green}
        print(f"scene randomised: cube is {COLOR_NAMES[colour]} (external node must classify it)")

def setup_ros_graph():
    """PC A의 ROS 2 Bridge 그래프를 한 번만 만든다.

    D455의 /rgb publisher는 로드한 USD의 /World/Graph/camera_graph에 이미 있다.
    여기서 또 만들면 /rgb publisher가 두 개가 되어 영상이 서로 번갈아 보인다.
    따라서 이 함수는 /color_id subscriber와 메시지 Counter만 추가한다.

    범용 ROS2Subscriber가 std_msgs/Int32를 받고 Counter가 새 메시지마다 증가한다.
    Python rclpy 노드를 만들지 않으므로 Python 3.11/3.12 충돌도 없다.
    """
    graph_path = "/ROS2_Color_Pick_Place"
    og.Controller.edit({"graph_path":graph_path,"evaluator_name":"execution"},{
        og.Controller.Keys.CREATE_NODES:[
            ("Tick", "omni.graph.action.OnPlaybackTick"),
            ("ColorId", "isaacsim.ros2.bridge.ROS2Subscriber"),
            ("ColorEventCounter", "omni.graph.action.Counter"),
        ],
        og.Controller.Keys.CONNECT:[
            ("Tick.outputs:tick", "ColorId.inputs:execIn"),
            ("ColorId.outputs:execOut", "ColorEventCounter.inputs:execIn"),
        ],
        og.Controller.Keys.SET_VALUES:[
            ("ColorId.inputs:topicName", "/color_id"),
            ("ColorId.inputs:messagePackage", "std_msgs"),
            ("ColorId.inputs:messageSubfolder", "msg"),
            ("ColorId.inputs:messageName", "Int32"),
        ]
    })
    # 범용 Subscriber는 messagePackage/messageName 설정 후 동적 data 출력 포트를
    # 만든다. 한 프레임 갱신해 포트가 생성된 뒤에 Python에서 그 값을 읽는다.
    simulation_app.update()
    return ColorIdReceiver(graph_path)

def init_gripper(robot,world):
    """Articulation이 준비된 뒤 ParallelGripper를 로봇 관절 제어에 연결한다."""
    # initialize의 첫 네 인자는 physics view가 아니라 함수/DOF 목록이다.
    # 키워드 인자를 사용해야 Isaac Sim 버전별 인자 순서 변화에도 안전하다.
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )
def ready(robot):
    """그리퍼가 아래를 향하는 알려진 시작 관절 자세로 되돌린다."""
    q=np.zeros(robot.num_dof); q[:6]=np.deg2rad(READY_JOINTS_DEG); robot.set_joint_positions(q)
def solver(robot):
    """URDF/descriptor 기반 Lula 역기구학 계산기를 생성한다."""
    l=LulaKinematicsSolver(robot_description_path=DESCRIPTION_PATH,urdf_path=URDF_PATH)
    l.set_robot_base_pose(robot_position=np.zeros(3),robot_orientation=np.array([1.,0,0,0]))
    return ArticulationKinematicsSolver(
        robot_articulation=robot,
        kinematics_solver=l,
        end_effector_frame_name=EE_LINK_NAME,
    )

def main():
    """프로그램 수명 주기와 Play/reset, ROS 수신, IK 제어를 관리한다."""
    ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "0")
    print(f"PC A ROS_DOMAIN_ID={ros_domain_id} (PC B must use the same value)")
    world=World(stage_units_in_meters=1.0); task=Task(); world.add_task(task); world.reset()
    color_receiver=setup_ros_graph()
    robot=task.robot; robot.initialize(); init_gripper(robot,world); ready(robot)
    ik=solver(robot); fsm=PickPlaceFSM(robot); q=target_quat(); was_playing=False
    restart_wait_steps=0
    print("Press Play. The robot will pick first, then wait for /color_id.")
    while simulation_app.is_running():
        world.step(render=True); time.sleep(.005); playing=world.is_playing()
        # Viewport에서 Play를 처음 누른 순간을 새 작업 주기로 취급한다.
        if playing and not was_playing:
            world.reset(); robot.initialize(); init_gripper(robot,world); ready(robot)
            task.randomize(); fsm.reset(); fsm.begin_pick(task.pick_xy)
            restart_wait_steps=0
        if playing:
            # 큐브를 집어 들어 올린 뒤부터 새로운 색상 메시지만 받는다.
            if fsm.state==fsm.WAIT:
                if not fsm.wait_initialized:
                    color_receiver.begin_cycle()
                    fsm.wait_initialized=True
                    fsm.wait_started=time.monotonic()
                    print(f"cube lifted; flushing stale /color_id for {COLOR_SETTLE_SEC}s")
                value=color_receiver.result()
                if value: fsm.set_destination(task.destinations[value])
                elif time.monotonic()-fsm.wait_started>=COLOR_TIMEOUT_SEC:
                    print(f"no valid /color_id yet; still waiting (ROS_DOMAIN_ID={ros_domain_id})")
                    fsm.wait_started=time.monotonic()
            # 큐브를 놓은 뒤에는 IK가 아닌 관절 공간 보간으로 READY 자세로 복귀한다.
            elif fsm.state==fsm.RETURN:
                arm_positions=fsm.return_arm_positions()
                robot.apply_action(ArticulationAction(
                    joint_positions=arm_positions,
                    joint_indices=np.arange(6),
                ))
                robot.apply_action(robot.gripper.forward(action="open"))
            # READY 자세 복귀가 끝나면 같은 객체를 새 위치/색으로 재배치하고
            # 다음 pick→검사→place 사이클을 자동으로 시작한다.
            elif fsm.state==fsm.DONE:
                restart_wait_steps+=1
                robot.apply_action(robot.gripper.forward(action="open"))
                if restart_wait_steps==1:
                    print(f"next cycle starts after {AUTO_RESTART_WAIT_STEPS} steps")
                if restart_wait_steps>=AUTO_RESTART_WAIT_STEPS:
                    task.randomize()
                    fsm.reset()
                    fsm.begin_pick(task.pick_xy)
                    restart_wait_steps=0
            # 나머지 구간은 TCP 목표 → 플랜지 목표 변환 → IK → 관절 명령 순서다.
            else:
                goal=fsm.current(); action,ok=ik.compute_inverse_kinematics(target_position=flange(goal,q),target_orientation=q)
                if ok: robot.apply_action(action)
                robot.apply_action(robot.gripper.forward(action=fsm.gripper)); fsm.advance()
        was_playing=playing
    simulation_app.close()
if __name__=="__main__": main()
