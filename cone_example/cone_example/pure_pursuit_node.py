import math
import numpy as np
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import LaserScan
from scipy.interpolate import CubicSpline
import rclpy
from rclpy.node import Node

# 최대 라이다 감지 거리
MAX_RANGE = 20.0
# 클러스터 분리 기준 거리 변화량
DELTA = 0.5
# Pure Pursuit 기본 lookahead 거리
LOOKAHEAD_DIST = 5.0
# 라이다 각도별 x, y 변환용 사인/코사인 배열 (차량 기준 전방이 y축)
COS = np.cos(np.linspace(0, 2*np.pi, 360, endpoint=False) + np.pi/2)
SIN = np.sin(np.linspace(0, 2*np.pi, 360, endpoint=False) + np.pi/2)
# 전방 180도 (좌우 90도씩) 인덱스 마스크
FRONT = (np.arange(360) <= 90) | (np.arange(360) >= 270)

class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        # 모터 명령 퍼블리셔 생성
        self.motor_pub = self.create_publisher(XycarMotor, 'xycar_motor', 10)
        # 라이다 데이터 구독자 생성
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, 10)
        # 이전 보간 경로 저장용
        self.prev_interp_x = None
        self.prev_interp_y = None

    def lidar_callback(self, msg):
        # 라이다 거리 데이터(360개) numpy 배열로 변환
        scan_data = np.array(msg.ranges[:360], dtype=np.float32)
        # 경로 생성 및 조향/속도 결정 함수 호출
        self.process_cone_pure_with_viz(scan_data)

    def grow_clusters(self, seed_list, all_clusters, threshold=2.5):
        # seed_list에서 시작해 threshold 이내에 있는 점들을 클러스터로 확장
        grown = seed_list.copy()          # 확장된 클러스터 결과
        queue = seed_list.copy()          # 탐색 대기열
        visited = set(seed_list)          # 이미 방문한 점 집합
        while queue:
            base = queue.pop(0)           # 대기열에서 하나 꺼냄
            bx, by = base                 # 기준점 좌표
            for pt in all_clusters:       # 모든 클러스터 후보에 대해
                if pt in visited:         # 이미 방문했다면 건너뜀
                    continue
                px, py = pt               # 후보점 좌표
                dist = np.hypot(px - bx, py - by)  # 기준점과의 거리 계산
                if dist <= threshold:     # threshold 이내면 클러스터에 추가
                    grown.append(pt)
                    queue.append(pt)      # 대기열에도 추가하여 계속 확장
                    visited.add(pt)
        return grown                      # 확장된 클러스터 반환

    def pure_pursuit(self, interp_x, interp_y, lookahead_dist):
        # 보간 경로(interp_x, interp_y)에서 lookahead 거리만큼 떨어진 목표점 찾기
        dists = np.sqrt(interp_x**2 + interp_y**2)     # 각 점까지의 거리
        candidates = np.where(dists > lookahead_dist)[0]  # lookahead 거리 넘는 인덱스
        if len(candidates) == 0:
            # 없으면 마지막 점을 목표로
            tx, ty = interp_x[-1], interp_y[-1]
        else:
            # lookahead 거리 넘는 첫 번째 점 중 차량 앞쪽(y>0)만 고려
            best_idx = candidates[0]
            for idx in candidates:
                if interp_y[idx] > 0:
                    best_idx = idx
                    break
            tx, ty = interp_x[best_idx], interp_y[best_idx]
        angle_rad = math.atan2(tx, ty)                 # 목표점에 대한 조향각(라디안)
        angle_deg = math.degrees(angle_rad)            # 각도를 도 단위로 변환
        angle_cmd = int(np.clip(angle_deg / 0.2, -100, 100))  # 조향 명령값(-100~100)
        return angle_cmd, (tx, ty)                     # 조향값, 목표점 반환

    def dynamic_lookahead_from_path(self, interp_x, interp_y, scale=0.2, min_ld=1.1, max_ld=5.0):
        # 경로 길이에 따라 lookahead 거리 동적으로 계산
        dx = np.diff(interp_x)                         # x 변화량
        dy = np.diff(interp_y)                         # y 변화량
        dists = np.sqrt(dx**2 + dy**2)                 # 각 구간의 거리
        total_length = np.sum(dists)                   # 전체 경로 길이
        dynamic_ld = np.clip(scale * total_length, min_ld, max_ld)  # scale%로, min/max 제한
        return dynamic_ld                              # lookahead 거리 반환

    def compute_speed(self, angle_cmd):
        # 조향각에 따라 속도 동적으로 결정
        MAX_SPEED = 20.0
        MIN_SPEED = 10.0
        angle_abs = abs(angle_cmd)                     # 조향 절대값
        norm = angle_abs / 100.0                       # 0~1 정규화
        speed = MAX_SPEED - (MAX_SPEED - MIN_SPEED) * norm  # 조향이 크면 속도 감소
        return int(speed)                              # 정수 속도 반환

    def process_cone_pure_with_viz(self, scan_data):
        # 라이다 데이터를 받아 경로 생성 및 조향/속도 결정
        motor_msg = XycarMotor()                       # 모터 명령 메시지 생성

        # 유효(무한대 아님, 최대거리 이내) & 전방 포인트 인덱스 추출
        valid = np.isfinite(scan_data) & (scan_data <= MAX_RANGE)
        idx = np.where(valid & FRONT)[0]
        if idx.size == 0:
            # 유효 포인트 없으면 멈춤
            self.motor_pub.publish(motor_msg)
            return

        # 라이다 거리 polar 좌표를 x, y로 변환
        x_all = scan_data * COS
        y_all = scan_data * SIN
        x, y = x_all[idx], y_all[idx]

        # 클러스터 분할: 인접하지 않거나 거리 변화가 큰 지점에서 분할
        diff_adj = np.diff(idx) != 1
        diff_r = np.abs(np.diff(scan_data[idx])) >= DELTA
        cuts = np.nonzero(diff_adj | diff_r)[0] + 1
        clusters = np.split(idx, cuts)
        # 각 클러스터의 중심점(x, y) 계산
        cluster_centers = [ (x_all[cl].mean(), y_all[cl].mean()) for cl in clusters ]

        # 좌/우 클러스터(콘) 중 차량 기준 가장 가까운(앞쪽) seed 찾기
        left_seed = min([pt for pt in cluster_centers if pt[0] < 0], key=lambda p: p[1], default=None)
        right_seed = min([pt for pt in cluster_centers if pt[0] > 0], key=lambda p: p[1], default=None)

        # 좌/우 클러스터 확장 및 가장 가까운 콘 찾기
        if left_seed:
            left_cones = self.grow_clusters([left_seed], cluster_centers)
        else:
            left_cones = []

        if right_seed:
            right_cones = self.grow_clusters([right_seed], cluster_centers)

        else:
            right_cones = []

        # 좌/우 콘 쌍의 중간점(midpoint) 계산
        midpoints = []
        for i in range(min(len(left_cones), len(right_cones))):
            if left_cones[i] == right_cones[i]:
                continue
            mx = (left_cones[i][0] + right_cones[i][0]) / 2
            my = (left_cones[i][1] + right_cones[i][1]) / 2
            midpoints.append((mx, my))

        if midpoints:
            # 중간점을 y(전방) 기준으로 정렬
            midpoints = sorted(midpoints, key=lambda p: p[1])
            mxs = np.array([p[0] for p in midpoints])
            mys = np.array([p[1] for p in midpoints])
            # y값 중복 제거(경로 보간 안정화)
            mys, unique_idx = np.unique(mys, return_index=True)
            mxs = mxs[unique_idx]

            if len(mxs) >= 2:
                # 2개 이상이면 CubicSpline으로 경로 보간
                spline_fn = CubicSpline(mys, mxs)
                interp_y = np.linspace(mys.min(), mys.max(), 100)
                interp_x = spline_fn(interp_y)
                # 보간 경로 저장(다음 프레임 fallback용)
                self.prev_interp_x, self.prev_interp_y = interp_x, interp_y
            elif self.prev_interp_x is not None:
                # 이전 경로 fallback
                interp_x, interp_y = self.prev_interp_x, self.prev_interp_y
            else:
                # 경로 생성 불가 시 멈춤
                angle_cmd = 0
                motor_msg.angle = angle_cmd
                motor_msg.speed = self.compute_speed(angle_cmd)
                self.motor_pub.publish(motor_msg)
                return
        else:
            # 중간점 자체가 없으면 멈춤
            angle_cmd = 0
            motor_msg.angle = angle_cmd
            motor_msg.speed = self.compute_speed(angle_cmd)
            self.motor_pub.publish(motor_msg)
            return

        # 경로 길이에 따라 lookahead 거리 동적 계산
        lookahead_dist = self.dynamic_lookahead_from_path(interp_x, interp_y)
        # Pure Pursuit로 조향각 및 목표점 계산
        angle_cmd, target = self.pure_pursuit(interp_x, interp_y, lookahead_dist)

        # 특수 상황: 좌측 콘만 1~3개, 우측 없음 → 속도/조향 고정 ->
        # ------------------- 수정 필요 ---------------
        if 1 <= len(left_cones) <= 3 and not right_cones:
            angle_cmd = 0
            motor_msg.speed = 50
        # ------------------- 수정 필요 --------------- (탈출 구간 트리거 만들면 됨)

        # 조향각/속도 적용 후 퍼블리시
        motor_msg.angle = angle_cmd
        motor_msg.speed = self.compute_speed(angle_cmd)
        self.motor_pub.publish(motor_msg)

def main(args=None):
    # ROS2 노드 초기화
    rclpy.init(args=args)
    # PurePursuitNode 인스턴스 생성
    node = PurePursuitNode()
    # 콜백 기반 이벤트 루프 실행
    rclpy.spin(node)
    # 종료 시 노드 해제 및 shutdown
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
