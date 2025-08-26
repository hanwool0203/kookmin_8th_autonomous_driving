import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String
from custom_interfaces.msg import ClusterData, ConeData

import numpy as np
from scipy.interpolate import CubicSpline

# --- 상수 정의 ---
SEED_MAX_DISTANCE = 1.0
LIDAR_TO_REAR_AXLE = 0.42

class PathPlanningNode(Node):
    def __init__(self):
        super().__init__('path_planning_node')
        self.is_active = False
        self.cluster_sub = self.create_subscription(
            ClusterData, '/clusters', self.cluster_callback, 10
        )
        self.path_pub = self.create_publisher(Path, '/path', 10)
        self.cone_pub = self.create_publisher(ConeData, '/cone_clusters', 10)
        self.midpoint_pub = self.create_publisher(ClusterData, '/midpoints', 10)
        self.activation_sub = self.create_subscription(String, '/sign_color', self.activation_callback, 10)
        self.prev_interp_x = None
        self.prev_interp_y = None

    def activation_callback(self, msg: String):
        if msg.data == 'green' and not self.is_active:
            self.get_logger().info('!!! path_planning activated by green signal !!!')
            self.is_active = True

    def cluster_callback(self, msg: ClusterData):
        
        if not self.is_active:
            return

        cluster_centers = [(p.x, p.y) for p in msg.clusters]
        
        # 1. 실제 감지된 클러스터로부터 초기 콘 그룹 형성
        left_cones, right_cones = self.form_cone_groups(cluster_centers)
        
        # 2. 필요 시 가상의 점을 명시적으로 추가
        if not left_cones and len(right_cones) > 1:
            v_cone = (0.1, 0.5)
            left_cones.append(v_cone)
        elif not right_cones and left_cones:
            v_cone = (0.1, -0.5)
            right_cones.append(v_cone)

        # 3. 시각화를 위해 최종 콘 목록(가상의 점 포함 가능)을 발행
        cone_msg = ConeData()
        cone_msg.left_cones = [Point(x=c[0], y=c[1], z=0.0) for c in left_cones]
        cone_msg.right_cones = [Point(x=c[0], y=c[1], z=0.0) for c in right_cones]
        self.cone_pub.publish(cone_msg)

        # 4. 최종 콘 목록을 기반으로 중간점 계산
        midpoints_lidar = self.calculate_midpoints_from_final_cones(left_cones, right_cones)
        
        # 5. 시각화를 위해 중간점 발행
        midpoint_msg = ClusterData()
        midpoint_msg.clusters = [Point(x=p[0], y=p[1], z=0.0) for p in midpoints_lidar]
        self.midpoint_pub.publish(midpoint_msg)

        # 6. 최종 경로 생성 및 발행
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "rear_axle"

        if len(midpoints_lidar) >= 2:
            rear_axle_midpoints = [(p[0] + LIDAR_TO_REAR_AXLE, p[1]) for p in midpoints_lidar]
            interp_x_ra, interp_y_ra = self.interpolate_path(rear_axle_midpoints)
            if interp_x_ra is not None:
                for x, y in zip(interp_x_ra, interp_y_ra):
                    pose = PoseStamped()
                    pose.pose.position.x = x
                    pose.pose.position.y = y
                    path_msg.poses.append(pose)
        elif len(midpoints_lidar) == 1:
            target_lidar = midpoints_lidar[0]
            pose = PoseStamped()
            pose.pose.position.x = target_lidar[0] + LIDAR_TO_REAR_AXLE
            pose.pose.position.y = target_lidar[1]
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)

    def calculate_midpoints_from_final_cones(self, left_cones, right_cones):
        midpoints = []
        if len(left_cones) >= 2 and len(right_cones) >= 2:
            midpoints = self.calculate_midpoints_pair(left_cones, right_cones)
        elif len(left_cones) == 1 and len(right_cones) > 1:
            midpoints = [((left_cones[0][0] + r[0])/2, (left_cones[0][1] + r[1])/2) for r in right_cones]
        elif len(right_cones) == 1 and len(left_cones) > 1:
            midpoints = [((l[0] + right_cones[0][0])/2, (l[1] + right_cones[0][1])/2) for l in left_cones]
        elif len(left_cones) == 1 and len(right_cones) == 1:
            l_cone, r_cone = left_cones[0], right_cones[0]
            midpoints.append(((l_cone[0] + r_cone[0]) / 2, (l_cone[1] + r_cone[1]) / 2))
        return midpoints
    
    def form_cone_groups(self, cluster_centers):
        # --- 부채꼴 영역 및 거리 상수 정의 ---
        LEFT_FIRST_CONE_RANGE = 0.8  
        RIGHT_FIRST_CONE_RANGE = 0.8
        LEFT_SECTOR_START, LEFT_SECTOR_END = 30.0, 100.0    # 좌측 부채꼴 각도 (degree)
        RIGHT_SECTOR_START, RIGHT_SECTOR_END = 260.0, 350.0  # 우측 부채꼴 각도 (degree)

        left_seed = None
        right_seed = None
        min_left_r = float('inf')
        min_right_r = float('inf')

        # 모든 클러스터를 순회하며 각 부채꼴 영역에서 가장 가까운 콘을 찾음
        for pt in cluster_centers:
            r = np.hypot(pt[0], pt[1])

            # 각도를 0-360도 범위로 계산
            theta_rad = np.arctan2(pt[1], pt[0])
            deg = np.rad2deg(theta_rad)
            if deg < 0:
                deg += 360.0

            # 좌측 부채꼴 영역 확인
            if LEFT_SECTOR_START <= deg <= LEFT_SECTOR_END and r <= LEFT_FIRST_CONE_RANGE:
                if r < min_left_r:
                    min_left_r = r
                    left_seed = pt
            # 우측 부채꼴 영역 확인
            elif RIGHT_SECTOR_START <= deg <= RIGHT_SECTOR_END and r <= RIGHT_FIRST_CONE_RANGE:
                if r < min_right_r:
                    min_right_r = r
                    right_seed = pt
        
        # 찾은 시드 콘을 기반으로 클러스터 확장
        used_clusters = set()
        left_cones = self.grow_clusters([left_seed], cluster_centers, used_clusters) if left_seed else []
        right_cones = self.grow_clusters([right_seed], cluster_centers, used_clusters) if right_seed else []
        
        return left_cones, right_cones

    # def form_cone_groups(self, cluster_centers):
    #     initial_left = [pt for pt in cluster_centers if pt[1] > 0]  # y > 0 이 좌측
    #     initial_right = [pt for pt in cluster_centers if pt[1] < 0]
    #     left_candidates = [pt for pt in initial_left if np.hypot(pt[0], pt[1]) < SEED_MAX_DISTANCE]
    #     right_candidates = [pt for pt in initial_right if np.hypot(pt[0], pt[1]) < SEED_MAX_DISTANCE]
    #     left_seed = min(left_candidates, key=lambda p: np.hypot(p[0], p[1]), default=None)
    #     right_seed = min(right_candidates, key=lambda p: np.hypot(p[0], p[1]), default=None)
    #     used_clusters = set()
    #     left_cones = self.grow_clusters([left_seed], cluster_centers, used_clusters) if left_seed else []
    #     right_cones = self.grow_clusters([right_seed], cluster_centers, used_clusters) if right_seed else []
    #     return left_cones, right_cones

    def grow_clusters(self, seed_list, all_clusters, used_clusters_set, threshold=0.5):
        if not seed_list or tuple(seed_list[0]) in used_clusters_set: return []
        grown, queue = [], []
        initial_seed_tuple = tuple(seed_list[0])
        grown.append(seed_list[0])
        queue.append(seed_list[0])
        used_clusters_set.add(initial_seed_tuple)
        head = 0
        while head < len(queue):
            base = queue[head]; head += 1
            for pt in all_clusters:
                pt_tuple = tuple(pt)
                if pt_tuple in used_clusters_set: continue
                if np.hypot(pt[0] - base[0], pt[1] - base[1]) <= threshold:
                    grown.append(pt); queue.append(pt); used_clusters_set.add(pt_tuple)
        return grown

    def calculate_midpoints_pair(self, left_cones, right_cones):
        midpoints = []
        MAX_VALID_DISTANCE = 1.3
        left_sorted = sorted(left_cones, key=lambda p: p[0])
        right_sorted = sorted(right_cones, key=lambda p: p[0])  
        if not left_sorted or not right_sorted: return []
        for l_cone in left_sorted:
            candidates = [(abs(l_cone[1] - r_cone[1]), r_cone) for r_cone in right_sorted]
            if not candidates: continue
            _, best_r_cone = min(candidates)
            dist = np.hypot(l_cone[0] - best_r_cone[0], l_cone[1] - best_r_cone[1])
            if dist > MAX_VALID_DISTANCE:
                continue
            mx, my = (l_cone[0] + best_r_cone[0]) / 2, (l_cone[1] + best_r_cone[1]) / 2
            midpoints.append((mx, my))
        return midpoints
        
    def interpolate_path(self, midpoints):
        if len(midpoints) < 2: return self.prev_interp_x, self.prev_interp_y
        midpoints = sorted(midpoints, key=lambda p: p[0]) # 1. 전방 거리(x) 기준으로 정렬
        mxs = np.array([p[0] for p in midpoints]); mys = np.array([p[1] for p in midpoints])
        mxs, unique_idx = np.unique(mxs, return_index=True); mys = mys[unique_idx]
        if len(mxs) < 2: return self.prev_interp_x, self.prev_interp_y
        spline_fn = CubicSpline(mxs, mys)
        interp_x = np.linspace(mxs.min(), mxs.max(), 100); interp_y = spline_fn(interp_x) 
        self.prev_interp_x, self.prev_interp_y = interp_x, interp_y
        return interp_x, interp_y

def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
