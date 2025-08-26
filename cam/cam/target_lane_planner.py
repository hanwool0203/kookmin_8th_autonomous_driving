import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from std_msgs.msg import String
from custom_interfaces.msg import Curve, Detections, ObstacleState, Ultrasonic
import message_filters
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import qos_profile_sensor_data
import yaml
from sklearn.cluster import DBSCAN
import os

class TargetLanePlanner(Node):
    def __init__(self):
        super().__init__('target_lane_planner')
        pkg_share = get_package_share_directory('cam')
        intrinsic_file = os.path.join(pkg_share, 'config', 'fisheye_calib.yaml')
        extrinsic_file = os.path.join(pkg_share, 'config', 'extrinsic.yaml')
        with open(intrinsic_file, 'r') as f: intrinsic_calib = yaml.safe_load(f)
        self.img_size = (intrinsic_calib['image_width'], intrinsic_calib['image_height'])
        self.mtx = np.array(intrinsic_calib['K'])
        with open(extrinsic_file, 'r') as f: extrinsic_calib = yaml.safe_load(f)
        self.R, self.t = np.array(extrinsic_calib['R']), np.array(extrinsic_calib['t']).reshape((3, 1))
        self.bridge = CvBridge()

        self.image_subscription = message_filters.Subscriber(self, Image, '/image_raw')
        self.curve_subscription = message_filters.Subscriber(self, Curve, '/center_curve')
        self.yolo_subscription = message_filters.Subscriber(self, Detections, '/yolo_detections')
        self.scan_subscription = message_filters.Subscriber(self, LaserScan, '/scan', qos_profile=qos_profile_sensor_data)
        self.ultra_subscription = message_filters.Subscriber(self, Ultrasonic, '/ultrasonic')

        self.obstacle_state_pub = self.create_publisher(ObstacleState, '/obstacle_state', 10)
        self.override_pub = self.create_publisher(String, '/lane_override_cmd', 10)
        
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.image_subscription, self.curve_subscription, self.yolo_subscription, self.scan_subscription, self.ultra_subscription],
            queue_size=10, slop=0.1)
        self.time_synchronizer.registerCallback(self.synchronized_callback)

        self.ULTAR_THRESHOLD = 30
        self.ULTAR_BACK_THRESHOLD = 40
        self.OVERTAKE_THRESHOLD_M = 1.95
        self.CURVE_OVERTAKE_THRESHOLD_M = 1.0
        self.MULTI_VEHICLE_OVERTAKE_GAP_M = 0.8
        self.driving_state = "CENTER_DRIVING"  # "CENTER_DRIVING" or "OVERTAKING"
        self.last_override_cmd = 'reset'

        self.is_passing_obstacle = False    # 장애물 옆을 통과하고 있는지 여부
        self.pass_complete_timer = None     # 장애물을 지나친 후 안전거리 확보를 위한 타이머
        self.PASS_TIMER_DURATION_S = 0.1  # 안전거리 확보 시간 (0.1초)

        self.E_STOP_DISTANCE_M = 0.5  # 급정지를 발동할 거리 (m)
        self.E_STOP_X_MIN = 50     # 급정지를 감지할 전방 카메라 x좌표 (최소)
        self.E_STOP_X_MAX = 600   # 급정지를 감지할 전방 카메라 x좌표 (최대)
        
        # ========================= 무조건 안 박음 =============================
        # self.E_STOP_RECT_X_MIN = 200
        # self.E_STOP_RECT_X_MAX = 400
        # self.E_STOP_RECT_Y_MIN = 260
        # self.E_STOP_RECT_Y_MAX = 400

        # # ========================= E-Stop 범위 작게 ==========================
        self.E_STOP_RECT_X_MIN = 280
        self.E_STOP_RECT_X_MAX = 360
        self.E_STOP_RECT_Y_MIN = 260
        self.E_STOP_RECT_Y_MAX = 400

        self.E_STOP_RECT_PIXEL_COUNT_THRESHOLD = 3

        self.PROXIMITY_THRESHOLD_M = 0.6
        self.smoothed_obstacle_position = 0.0  # -1 (left) ~ +1 (right) 사이의 값을 가짐
        self.EMA_ALPHA = 0.3  # 스무딩 강도 (0.1: 매우 부드러움, 0.9: 매우 민감)
        self.last_closest_obstacle_lane = "unknown" # 가장 가까운 장애물의 마지막 차선 위치 저장

        # 차선 판단 로직 임계값 변수
        self.AREA_CLIP_THRESHOLD_PERCENT = 13.0
        self.LANE_DETERMINATION_DISTANCE_M = 1.95 # 차선 판단을 시작할 최대 거리

        self.get_logger().info('🎯 Target Lane Planner has been started!')
    
    def _find_intersections(self, curve_points, xmin, ymin, xmax, ymax):
        """커브(폴리라인)와 바운딩 박스의 교차점을 찾는 헬퍼 함수"""
        intersection_points = []
        box_segments = [
            ((xmin, ymin), (xmax, ymin)),  # Top
            ((xmin, ymax), (xmax, ymax)),  # Bottom
            ((xmin, ymin), (xmin, ymax)),  # Left
            ((xmax, ymin), (xmax, ymax))   # Right
        ]

        for i in range(len(curve_points) - 1):
            p1 = curve_points[i]
            p2 = curve_points[i+1]
            x1, y1 = p1
            x2, y2 = p2

            for seg in box_segments:
                p3, p4 = seg
                x3, y3 = p3
                x4, y4 = p4

                den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if den == 0:
                    continue

                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

                if 0 <= t <= 1 and 0 <= u <= 1:
                    ix = x1 + t * (x2 - x1)
                    iy = y1 + t * (y2 - y1)
                    intersection_points.append(np.array([ix, iy]))
        
        # 중복 제거 및 y좌표 기준 정렬
        if not intersection_points:
            return []
            
        unique_points = np.unique(np.array(intersection_points).round(decimals=2), axis=0)
        sorted_points = sorted(unique_points, key=lambda p: p[1])
        return sorted_points

    def project_lidar_on_image(self, image, detections_msg, points_3d, curve_msg,left_ultra,right_ultra,left_back_ultra,right_back_ultra,img_pts):
        # 1. 모든 장애물 정보 수집 (기존과 동일)
        processed_obstacles = []
        detections = [det for det in detections_msg.detections if det.class_name == 'obstacle_vehicle']

        h, w, _ = image.shape
        if points_3d.shape[0] == 0: img_pts = np.array([])
        else:
            cam_pts = (self.R @ points_3d.T + self.t).T
            img_pts, _ = cv2.projectPoints(cam_pts, np.zeros(3), np.zeros(3), self.mtx, None)
            img_pts = img_pts.squeeze(axis=1)
        has_curve = curve_msg and len(curve_msg.points) >= 2

        if has_curve:
            raw_curve_points = np.array([[p.x, p.y] for p in curve_msg.points])
            sorted_indices = np.argsort(raw_curve_points[:, 1])
            curve_points = raw_curve_points[sorted_indices]
            curve_x_coords = curve_points[:, 0]
            curve_y_coords = curve_points[:, 1]

        for det in detections:
            xmin, ymin, xmax, ymax = map(int, [det.xmin, det.ymin, det.xmax, det.ymax])
            avg_distance = -1.0
            points_in_path_count = 0
            area_display_str = "N/A"

            # ================================ LiDAR 포인트 클라우드 클러스터링 ================================
            if img_pts.shape[0] > 0:
                indices = np.where((img_pts[:, 0] >= xmin) & (img_pts[:, 0] < xmax) & (img_pts[:, 1] >= ymin) & (img_pts[:, 1] < ymax))[0]
                if len(indices) > 2:
                    points_in_bbox = points_3d[indices]
                    dbscan = DBSCAN(eps=0.3, min_samples=3)
                    clusters = dbscan.fit_predict(points_in_bbox)
                    img_pts_in_bbox = img_pts[indices]
                    unique_labels = np.unique(clusters[clusters != -1])
                    if len(unique_labels) > 0:
                        cluster_info_list = []
                        for label in unique_labels:
                            current_cluster_points_3d = points_in_bbox[clusters == label]
                            ranges = np.sqrt(current_cluster_points_3d[:, 0]**2 + current_cluster_points_3d[:, 1]**2)
                            avg_dist = np.mean(ranges)
                            cluster_info_list.append({'label': label, 'distance': avg_dist})
                        closest_cluster_info = min(cluster_info_list, key=lambda x: x['distance'])
                        closest_cluster_label = closest_cluster_info['label']
                        main_cluster_points = points_in_bbox[clusters == closest_cluster_label]
                        final_ranges = np.sqrt(main_cluster_points[:, 0]**2 + main_cluster_points[:, 1]**2)
                        avg_distance = np.mean(final_ranges)
                        img_pts_in_cluster = img_pts[indices][clusters == closest_cluster_label]
                        points_in_path = [pt for pt in img_pts_in_cluster if self.E_STOP_X_MIN < pt[0] < self.E_STOP_X_MAX]
                        points_in_path_count = len(points_in_path)
                        points_to_draw = img_pts_in_bbox[clusters == closest_cluster_label]
                        for pt in points_to_draw:
                            cv2.circle(image, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1) 
                    else:
                        self.get_logger().warn("DBSCAN failed to find a cluster. Using all points in bbox as fallback.")
                        if points_in_bbox.shape[0] > 0:
                            all_ranges = np.sqrt(points_in_bbox[:, 0]**2 + points_in_bbox[:, 1]**2)
                            avg_distance = np.mean(all_ranges)
                            for pt in img_pts_in_bbox:
                                cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (128, 128, 128), -1)
            # ================================ LiDAR 포인트 클라우드 클러스터링 ================================

            # ================================ 장애물 차량 State 판별 (수정된 로직) ================================
            position_str_raw = "unknown"
            if 0 < avg_distance <= self.LANE_DETERMINATION_DISTANCE_M and has_curve and len(curve_points) >= 2:
                intersections = self._find_intersections(curve_points, xmin, ymin, xmax, ymax)

                if len(intersections) >= 1:
                    p_entry = intersections[0]
                    p_exit = intersections[-1]
                    epsilon = 1.0 
                    box_width = xmax - xmin
                    box_height = ymax - ymin
                    is_wide_bbox = box_width > box_height
                    is_side_penetration = False
                    for p in [p_entry, p_exit]:
                        if abs(p[0] - xmin) < epsilon or abs(p[0] - xmax) < epsilon:
                            is_side_penetration = True
                            break
                    is_top_bottom_only = not is_side_penetration
                    penetrates_bottom_edge = (abs(p_entry[1] - ymax) < epsilon) or (abs(p_exit[1] - ymax) < epsilon)
                    is_significant_penetration = True
                    if box_width > 1 and box_height > 1:
                        total_area = box_width * box_height
                        calculated_area = -1
                        entry_on = [abs(p_entry[1] - ymin) < epsilon, abs(p_entry[1] - ymax) < epsilon, abs(p_entry[0] - xmin) < epsilon, abs(p_entry[0] - xmax) < epsilon]
                        exit_on = [abs(p_exit[1] - ymin) < epsilon, abs(p_exit[1] - ymax) < epsilon, abs(p_exit[0] - xmin) < epsilon, abs(p_exit[0] - xmax) < epsilon]
                        if (entry_on[0] and exit_on[2]) or (entry_on[2] and exit_on[0]):
                            p_top = p_entry if entry_on[0] else p_exit
                            p_left = p_exit if entry_on[0] else p_entry
                            calculated_area = 0.5 * abs(p_top[0] - xmin) * abs(p_left[1] - ymin)
                        elif (entry_on[0] and exit_on[3]) or (entry_on[3] and exit_on[0]):
                            p_top = p_entry if entry_on[0] else p_exit
                            p_right = p_exit if entry_on[0] else p_entry
                            calculated_area = 0.5 * abs(p_top[0] - xmax) * abs(p_right[1] - ymin)
                        elif (entry_on[1] and exit_on[2]) or (entry_on[2] and exit_on[1]):
                            p_bot = p_entry if entry_on[1] else p_exit
                            p_left = p_exit if entry_on[1] else p_entry
                            calculated_area = 0.5 * abs(p_bot[0] - xmin) * abs(p_left[1] - ymax)
                        elif (entry_on[1] and exit_on[3]) or (entry_on[3] and exit_on[1]):
                            p_bot = p_entry if entry_on[1] else p_exit
                            p_right = p_exit if entry_on[1] else p_entry
                            calculated_area = 0.5 * abs(p_bot[0] - xmax) * abs(p_right[1] - ymax)
                        elif (entry_on[2] and exit_on[3]) or (entry_on[3] and exit_on[2]): 
                            trapezoid_area = 0.5 * ((ymax - p_entry[1]) + (ymax - p_exit[1])) * box_width
                            calculated_area = min(trapezoid_area, total_area - trapezoid_area)
                        elif (entry_on[0] and exit_on[1]) or (entry_on[1] and exit_on[0]):
                            trapezoid_area = 0.5 * ((xmax - p_entry[0]) + (xmax - p_exit[0])) * box_height
                            calculated_area = min(trapezoid_area, total_area - trapezoid_area)
                        if calculated_area >= 0:
                            area_percent = (calculated_area / total_area) * 100
                            area_display_str = f"{area_percent:.1f}%"
                            if area_percent < self.AREA_CLIP_THRESHOLD_PERCENT: is_significant_penetration = False
                    dx = p_exit[0] - p_entry[0]
                    dy = p_exit[1] - p_entry[1]
                    penetrating_slope = dy / dx if abs(dx) > 1e-6 else float('inf')
                    is_vertical_line = penetrating_slope == float('inf')
                    use_slope_logic = not is_top_bottom_only and is_significant_penetration and is_wide_bbox and not is_vertical_line and not penetrates_bottom_edge
                    
                    # ========================= [수정된 부분 시작] ========================
                    # use_slope_logic이 True여도, 거리가 0.8m 이하이면 강제로 점 기반 판단으로 변경
                    if use_slope_logic and avg_distance > 0.8:
                    # ========================= [수정된 부분 끝] ==========================
                        self.get_logger().info(f"판단: 기울기 기반. Slope:{penetrating_slope:.2f}, Dist:{avg_distance:.2f}m")
                        position_str_raw = "right" if penetrating_slope < 0 else "left"
                        cv2.line(image, tuple(p_entry.astype(int)), tuple(p_exit.astype(int)), (0, 255, 255), 2)
                        if penetrating_slope >= 0: p_diag1, p_diag2 = (xmin, ymin), (xmax, ymax)
                        else: p_diag1, p_diag2 = (xmax, ymin), (xmin, ymax)
                        cv2.line(image, p_diag1, p_diag2, (255, 0, 255), 1)
                    else:
                        if penetrates_bottom_edge: self.get_logger().info(f"판단: 점 비교 (하단면 관통). Dist:{avg_distance:.2f}m")
                        elif not use_slope_logic: self.get_logger().info(f"판단: 점 비교 (기타). Dist:{avg_distance:.2f}m")
                        else: self.get_logger().info(f"판단: 점 비교 (거리 < 0.8m 강제). Dist:{avg_distance:.2f}m")
                        
                        bbox_center_x = (xmin + xmax) / 2
                        bbox_center_y = (ymin + ymax) / 2

                        path_comparison_point = None
                        # [예외 상황] BBox 중심이 경로 Y 범위 밖 (외삽 발생 구간)
                        if len(curve_y_coords) < 2 or bbox_center_y < curve_y_coords[0] or bbox_center_y > curve_y_coords[-1]:
                            self.get_logger().warn("BBox Y is outside curve range. Extending FIRST path segment to side boundaries.")
                            p0, p1 = curve_points[0], curve_points[1] # 경로의 '첫 번째'와 '두 번째' 점
                            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                            if abs(dx) < 1e-6: # 수직선 예외 처리
                                path_comparison_point = p1
                            else:
                                m = dy / dx
                                c = p1[1] - m * p1[0]
                                y_at_x0 = c
                                y_at_xw = m * (w - 1) + c
                                intersect_p_left = np.array([0, y_at_x0])
                                intersect_p_right = np.array([w - 1, y_at_xw])
                                dist_left = abs(intersect_p_left[0] - p0[0])
                                dist_right = abs(intersect_p_right[0] - p0[0])
                                if dist_left <= dist_right:
                                    path_comparison_point = intersect_p_left
                                else:
                                    path_comparison_point = intersect_p_right
                        # [일반 상황] BBox 중심이 경로 Y 범위 안 (안전한 보간 구간)
                        else:
                            path_x_at_bbox_y = np.interp(bbox_center_y, curve_y_coords, curve_x_coords)
                            path_comparison_point = np.array([path_x_at_bbox_y, bbox_center_y])
                        
                        position_str_raw = "right" if bbox_center_x > path_comparison_point[0] else "left"
                        cv2.circle(image, (int(bbox_center_x), int(bbox_center_y)), 5, (0, 0, 255), -1)
                        cv2.circle(image, tuple(path_comparison_point.astype(int)), 5, (255, 255, 0), -1)
                        cv2.line(image, (int(bbox_center_x), int(bbox_center_y)), tuple(path_comparison_point.astype(int)), (255, 255, 255), 1)

                else:
                    area_display_str = "No Intersection"
                    self.get_logger().info("판단: 점 비교 (교차점 부족).")
                    bbox_center_x = (xmin + xmax) / 2
                    bbox_center_y = (ymin + ymax) / 2

                    # 교차점 없을 때도 외삽 문제 해결 로직 동일하게 적용
                    path_comparison_point = None
                    if len(curve_y_coords) < 2 or bbox_center_y < curve_y_coords[0] or bbox_center_y > curve_y_coords[-1]:
                        self.get_logger().warn("BBox Y is outside curve range. Extending FIRST path segment to side boundaries.")
                        p0, p1 = curve_points[0], curve_points[1]
                        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                        if abs(dx) < 1e-6:
                            path_comparison_point = p1
                        else:
                            m = dy / dx
                            c = p1[1] - m * p1[0]
                            y_at_x0 = c
                            y_at_xw = m * (w - 1) + c
                            intersect_p_left = np.array([0, y_at_x0])
                            intersect_p_right = np.array([w - 1, y_at_xw])
                            dist_left = abs(intersect_p_left[0] - p0[0])
                            dist_right = abs(intersect_p_right[0] - p0[0])
                            if dist_left <= dist_right:
                                path_comparison_point = intersect_p_left
                            else:
                                path_comparison_point = intersect_p_right
                    else:
                        path_x_at_bbox_y = np.interp(bbox_center_y, curve_y_coords, curve_x_coords)
                        path_comparison_point = np.array([path_x_at_bbox_y, bbox_center_y])

                    position_str_raw = "right" if bbox_center_x > path_comparison_point[0] else "left"
                    cv2.circle(image, (int(bbox_center_x), int(bbox_center_y)), 5, (0, 0, 255), -1)
                    cv2.circle(image, tuple(path_comparison_point.astype(int)), 5, (255, 255, 0), -1)
                    cv2.line(image, (int(bbox_center_x), int(bbox_center_y)), tuple(path_comparison_point.astype(int)), (255, 255, 255), 1)


            if avg_distance > 0:
                obstacle_data = {
                    'distance': avg_distance, 'position_raw': position_str_raw,
                    'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                    'points_in_path_count': points_in_path_count,
                    'area_ratio_str': area_display_str
                }
                processed_obstacles.append(obstacle_data)
                cv2.putText(image, f"{avg_distance:.1f}m", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, f"Raw: {position_str_raw}", (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        final_position_str = "unknown"
        target_obstacle_for_decision = None

        if processed_obstacles:
            target_obstacle_for_decision = min(processed_obstacles, key=lambda x: x['distance'])
            raw_pos_of_target = target_obstacle_for_decision['position_raw']
            
            # 차선 판단 로직이 수행되지 않아 'unknown'인 경우, 거리를 확인하여 fallback 여부 결정
            if raw_pos_of_target == "unknown":
                # 장애물이 판단 유효 거리 내에 있을 때만 fallback 로직(이전 값 사용)을 적용
                if target_obstacle_for_decision['distance'] <= self.LANE_DETERMINATION_DISTANCE_M:
                    raw_pos_of_target = self.last_closest_obstacle_lane
                    self.get_logger().warn(f"Lane determination failed for a CLOSE obstacle ({target_obstacle_for_decision['distance']:.2f}m), using last known lane: '{raw_pos_of_target}'")
                # 거리가 멀면 'unknown'을 그대로 유지 (fallback 없음)
            else:
                # 새로운 차선 정보가 유효하면(left/right), 최신 정보로 업데이트
                self.last_closest_obstacle_lane = raw_pos_of_target

            current_numeric_pos = 0.0
            if raw_pos_of_target == "left": current_numeric_pos = -1.0
            elif raw_pos_of_target == "right": current_numeric_pos = 1.0
            self.smoothed_obstacle_position = (self.EMA_ALPHA * current_numeric_pos) + ((1 - self.EMA_ALPHA) * self.smoothed_obstacle_position)
            if self.smoothed_obstacle_position < -0.1: final_position_str = "left"
            elif self.smoothed_obstacle_position > 0.1: final_position_str = "right"
            target_obstacle_for_decision['position_final'] = final_position_str
            self.get_logger().info(f"Closest Obstacle Raw: {raw_pos_of_target} -> Smoothed Val: {self.smoothed_obstacle_position:.2f} -> Final Decision: {final_position_str}")
        else:
            self.smoothed_obstacle_position = 0.0

        # ================================ E_STOP ================================
        cv2.rectangle(image, (self.E_STOP_RECT_X_MIN, self.E_STOP_RECT_Y_MIN), (self.E_STOP_RECT_X_MAX, self.E_STOP_RECT_Y_MAX), (0, 0, 255), 1)
        is_pixel_emergency = False
        if img_pts.shape[0] > 0 and len(detections) > 0:
            points_in_danger_zone = img_pts[
                (img_pts[:, 0] > self.E_STOP_RECT_X_MIN) & (img_pts[:, 0] < self.E_STOP_RECT_X_MAX) &
                (img_pts[:, 1] > self.E_STOP_RECT_Y_MIN) & (img_pts[:, 1] < self.E_STOP_RECT_Y_MAX)
            ]
            if len(points_in_danger_zone) > 0:
                for det in detections:
                    xmin, ymin, xmax, ymax = int(det.xmin), int(det.ymin), int(det.xmax), int(det.ymax)
                    points_in_both_zones = points_in_danger_zone[
                        (points_in_danger_zone[:, 0] > xmin) & (points_in_danger_zone[:, 0] < xmax) &
                        (points_in_danger_zone[:, 1] > ymin) & (points_in_danger_zone[:, 1] < ymax)
                    ]
                    if len(points_in_both_zones) > self.E_STOP_RECT_PIXEL_COUNT_THRESHOLD:
                        is_pixel_emergency = True
                        self.get_logger().error(f"🚨 RECT E-STOP TRIGGER! Found {len(points_in_both_zones)} points in both E-Stop Rect and Vehicle BBox.")
                        cv2.rectangle(image, (self.E_STOP_RECT_X_MIN, self.E_STOP_RECT_Y_MIN), (self.E_STOP_RECT_X_MAX, self.E_STOP_RECT_Y_MAX), (0, 0, 255), 3)
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                        break

        is_obstacle_emergency = False
        emergency_obstacle = None
        for obs in processed_obstacles:
            obs_position = obs['position_raw'] 
            is_center_path = self.last_override_cmd == 'reset'
            is_left_path = self.last_override_cmd == 'go_left'
            is_right_path = self.last_override_cmd == 'go_right'
            collision_path = (obs_position == 'left' and (is_center_path or is_left_path)) or \
                                (obs_position == 'right' and (is_center_path or is_right_path))
            if obs['distance'] < self.E_STOP_DISTANCE_M and obs['points_in_path_count'] > 0 and collision_path:
                is_obstacle_emergency = True
                emergency_obstacle = obs
                self.get_logger().error(f"🚨 E-STOP TRIGGER! Obstacle at {obs['distance']:.2f}m on a collision path.")
                break
        # ================================ E_STOP ================================

        # ================================ 추월 로직 시작 ================================
        obstacle_msg = ObstacleState()
        target_obstacle_for_state_machine = target_obstacle_for_decision
        current_override_cmd = self.last_override_cmd
        num_detected_vehicles = len(detections)
        is_curve = curve_msg and curve_msg.state.lower() == 'curve'

        # ================================= 추월용 =========================================

        # if self.driving_state == "CENTER_DRIVING":
        #     if num_detected_vehicles >= 2:
        #         self.driving_state = "Time_Gap"
        #         current_override_cmd = 'reset'
        #         self.get_logger().warn("STATE CHANGE: CENTER_DRIVING -> Time_Gap (2+ vehicles).")
        #     elif num_detected_vehicles == 1 and target_obstacle_for_state_machine:
        #         obstacle = target_obstacle_for_state_machine
        #         obstacle = target_obstacle_for_state_machine
        #         # [수정] 장애물 위치가 'unknown'이 아닐 경우에만 추월 상태로 진입하도록 조건 추가
        #         if not is_curve and obstacle['distance'] < self.OVERTAKE_THRESHOLD_M and obstacle['position_final'] != 'unknown':
        #             self.driving_state = "OVERTAKING"
        #             self.is_passing_obstacle = False
        #             self.pass_complete_timer = None
        #             current_override_cmd = 'go_left' if obstacle['position_final'] == 'right' else 'go_right'
        #             self.get_logger().warn(f"STATE CHANGE (Straight): CENTER_DRIVING -> OVERTAKING ({current_override_cmd}).")
        #         # [수정] 'unknown'인 경우, 중앙 주행 유지
        #         elif is_curve or obstacle['position_final'] == 'unknown':
        #             self.driving_state = "Time_Gap"
        #             current_override_cmd = 'reset'
        #             if is_curve:
        #                 self.get_logger().warn("STATE CHANGE (Curve): CENTER_DRIVING -> Time_Gap. Start following.")
        #             else:
        #                 self.get_logger().warn(f"STATE CHANGE (Unknown Pos): CENTER_DRIVING -> Time_Gap. Position is '{obstacle['position_final']}'. Following.")
        #         else:
        #             current_override_cmd = 'reset'
        #     else:
        #         current_override_cmd = 'reset'

        # elif self.driving_state == "OVERTAKING":
        #     current_override_cmd = self.last_override_cmd
        #     if not self.is_passing_obstacle:
        #         is_side_detected = (self.last_override_cmd == 'go_left' and right_ultra < self.ULTAR_THRESHOLD) or \
        #                            (self.last_override_cmd == 'go_right' and left_ultra < self.ULTAR_THRESHOLD)
        #         if is_side_detected:
        #             self.is_passing_obstacle = True
        #             self.get_logger().info("Side of obstacle detected. Now passing alongside.")
        #     else:
        #         is_side_cleared = (self.last_override_cmd == 'go_left' and right_ultra > self.ULTAR_THRESHOLD and right_back_ultra > self.ULTAR_BACK_THRESHOLD) or \
        #                           (self.last_override_cmd == 'go_right' and left_ultra > self.ULTAR_THRESHOLD and left_back_ultra > self.ULTAR_BACK_THRESHOLD)
        #         if is_side_cleared and self.pass_complete_timer is None:
        #             self.pass_complete_timer = self.get_clock().now()
        #             self.get_logger().info(f"Side of obstacle cleared. Starting {self.PASS_TIMER_DURATION_S}s safety timer.")
        #         if self.pass_complete_timer is not None:
        #             duration = self.get_clock().now() - self.pass_complete_timer
        #             if duration.nanoseconds / 1e9 > self.PASS_TIMER_DURATION_S:
        #                 self.driving_state = "CENTER_DRIVING"
        #                 current_override_cmd = 'reset'
        #                 self.get_logger().warn("STATE CHANGE: OVERTAKING -> CENTER_DRIVING (Overtake complete).")
        #                 self.is_passing_obstacle = False
        #                 self.pass_complete_timer = None

        # elif self.driving_state == "Time_Gap":
        #     if is_curve and num_detected_vehicles == 1 and target_obstacle_for_state_machine:
        #         if target_obstacle_for_state_machine['distance'] <= self.CURVE_OVERTAKE_THRESHOLD_M and target_obstacle_for_state_machine['position_final'] != 'unknown':
        #             self.driving_state = "OVERTAKING"
        #             self.is_passing_obstacle = False
        #             self.pass_complete_timer = None
        #             current_override_cmd = 'go_left' if target_obstacle_for_state_machine['position_final'] == 'right' else 'go_right'
        #             self.get_logger().warn(f"STATE CHANGE (Curve): Time_Gap -> OVERTAKING. Target close ({target_obstacle_for_state_machine['distance']:.2f}m).")
        #         else:
        #             current_override_cmd = 'reset'
        #             self.get_logger().info(f"Time_Gap (Curve): Following target at {target_obstacle_for_state_machine['distance']:.2f}m.")
        #     elif num_detected_vehicles >= 2 and len(processed_obstacles) >= 2:
        #         sorted_obstacles = sorted(processed_obstacles, key=lambda obs: obs['distance'])
        #         closest_car = sorted_obstacles[0]
        #         second_closest_car = sorted_obstacles[1]
        #         distance_diff = second_closest_car['distance'] - closest_car['distance']
        #         if distance_diff >= self.MULTI_VEHICLE_OVERTAKE_GAP_M and target_obstacle_for_decision['position_final'] != 'unknown':
        #             self.driving_state = "OVERTAKING"
        #             self.is_passing_obstacle = False
        #             self.pass_complete_timer = None
        #             current_override_cmd = 'go_left' if target_obstacle_for_decision['position_final'] == 'right' else 'go_right'
        #             self.get_logger().warn(f"STATE CHANGE (Multi-Vehicle): Time_Gap -> OVERTAKING. Gap is sufficient ({distance_diff:.2f}m >= {self.MULTI_VEHICLE_OVERTAKE_GAP_M}m).")
        #         else:
        #             current_override_cmd = 'reset'
        #             self.get_logger().info(f"Time_Gap (Multi-Vehicle): Gap is tight ({distance_diff:.2f}m < {self.MULTI_VEHICLE_OVERTAKE_GAP_M}m). Following closest car.")
        #     elif (not is_curve and num_detected_vehicles < 2) or num_detected_vehicles == 0:
        #         self.driving_state = "CENTER_DRIVING"
        #         current_override_cmd = 'reset'
        #         self.get_logger().warn("STATE CHANGE: Time_Gap -> CENTER_DRIVING (Exit condition met).")
        #     else:
        #         current_override_cmd = 'reset'
        #         if processed_obstacles:
        #             self.get_logger().info(f"Time_Gap: Following closest vehicle at {min(processed_obstacles, key=lambda obs: obs['distance'])['distance']:.1f}m.")
        #         else:
        #             self.get_logger().info("Time_Gap: Vehicle(s) detected, but no distance info. Driving straight.")
        # # =======================> 추월 가능 로직 끝 <=========================

        # ============================ 밑에는 확실한 곡선 추월 금지 로직 ======================
        # --------------- [전환 0] (Curve, 1), (Curve, 2) 의 경우 ---------------
        # --------------- 무조건 가장 가까운 차의 차선으로 변경 -> 측면 추돌 예방 -------
        # -- 발생 문제 : centerlane_tracer에서 Curve로 튀면 -> (Curve, 1)이 되어 버려서 추월하다가 장애물 차량의 차선으로 변경하여 측면 추돌해버릴 수도 있음. -----
        if is_curve and num_detected_vehicles >= 1 :
            self.driving_state = "Time_Gap"
            if processed_obstacles:
                target_obstacle_for_state_machine = min(processed_obstacles, key=lambda obs: obs['distance'])
                current_override_cmd = 'reset'
                self.get_logger().warn(f"(Curve, 1), (Curve, 2) Case : Target at {target_obstacle_for_state_machine['distance']:.1f}m.")

        else:
        # ----------- 나머지 (Straight, 0), (Straight, 1), (Straight, 2), (Curve, 0) 의 경우를 결정 --------
            # [상태 1] 기본 (CENTER_DRIVING)
            if self.driving_state == "CENTER_DRIVING":

                # --------------- [전환 1] (Straight, 2) 의 경우 ------------------
                # --------------- 앞에 무조건 직선 구간에 두 대가 있으므로 측면 추돌을 생각할 필요가 X -> center로 달려도 됨.-----
                if num_detected_vehicles >= 2:
                    self.driving_state = "Time_Gap"
                    self.get_logger().warn("STATE CHANGE: CENTER_DRIVING -> Time_Gap + (Straight, 2).")
                    if processed_obstacles:
                        target_obstacle_for_state_machine = min(processed_obstacles, key=lambda obs: obs['distance'])
                    current_override_cmd = 'reset'

                # --------------- [전환 2] (Straight, 1) 의 경우 ------------------
                elif num_detected_vehicles == 1 and processed_obstacles:
                    obstacle = processed_obstacles[0]
                    if obstacle['distance'] < self.OVERTAKE_THRESHOLD_M:
                        self.driving_state = "OVERTAKING"
                        self.is_passing_obstacle = False
                        self.pass_complete_timer = None
                        target_obstacle_for_state_machine = obstacle
                        current_override_cmd = 'go_left' if target_obstacle_for_state_machine['position_final'] == 'right' else 'go_right' # 장애물 차량과 반대 차선으로 차선 변경
                        self.get_logger().warn(f"STATE CHANGE: CENTER_DRIVING -> OVERTAKING ({current_override_cmd}).")
                    else: # 1대 있지만 멀리 있으면 중앙 주행 유지
                        current_override_cmd = 'reset'

                # --------------- [전환 3] (Straight, 0), (Curve, 0) 의 경우 ------------------       
                else: # 감지된 차량이 없거나, 있어도 거리 측정을 못하면 중앙 주행 유지
                    current_override_cmd = 'reset'

            # [상태 2] 추월 중 (OVERTAKING) + (Straight, 1)의 경우
            elif self.driving_state == "OVERTAKING":
                if processed_obstacles:
                    target_obstacle_for_state_machine = min(processed_obstacles, key=lambda obs: obs['distance'])

                current_override_cmd = self.last_override_cmd # 추월 중에는 차선 변경 명령 유지

                # 1단계: 아직 장애물 옆을 지나치기 시작하지 않았을 때
                if not self.is_passing_obstacle:
                    is_side_detected = (self.last_override_cmd == 'go_left' and right_ultra < self.ULTAR_THRESHOLD) or \
                                       (self.last_override_cmd == 'go_right' and left_ultra < self.ULTAR_THRESHOLD)
                    if is_side_detected:
                        self.is_passing_obstacle = True
                        self.get_logger().info("Side of obstacle detected. Now passing alongside.")
                    else:
                        self.get_logger().info(f"Approaching side... L: {left_ultra}, R: {right_ultra}, LB: {left_back_ultra}, RB: {right_back_ultra}")
                
                # 2단계: 장애물 옆을 지나치고 있을 때
                else:
                    is_side_cleared = (self.last_override_cmd == 'go_left' and right_ultra > self.ULTAR_THRESHOLD and right_back_ultra > self.ULTAR_BACK_THRESHOLD) or \
                                      (self.last_override_cmd == 'go_right' and left_ultra > self.ULTAR_THRESHOLD and left_back_ultra > self.ULTAR_BACK_THRESHOLD)

                    # 3단계: 장애물 끝을 통과하여 타이머를 시작해야 할 때
                    if is_side_cleared and self.pass_complete_timer is None:
                        self.pass_complete_timer = self.get_clock().now()
                        self.get_logger().info(f"Side of obstacle cleared. Starting {self.PASS_TIMER_DURATION_S}s safety timer.")
                    
                    # 타이머가 시작되었다면, 시간이 다 지났는지 확인
                    if self.pass_complete_timer is not None:
                        duration = self.get_clock().now() - self.pass_complete_timer
                        # 4단계: 타이머 종료 -> 추월 완료!
                        if duration.nanoseconds / 1e9 > self.PASS_TIMER_DURATION_S:
                            self.driving_state = "CENTER_DRIVING"
                            current_override_cmd = 'reset'
                            self.get_logger().warn("STATE CHANGE: OVERTAKING -> CENTER_DRIVING (Timer complete).")
                            # 상태 변수 초기화
                            self.is_passing_obstacle = False
                            self.pass_complete_timer = None
                        else:
                            self.get_logger().info(f"Safety timer running... ({duration.nanoseconds / 1e9:.2f}s)")
                    # 아직 장애물 옆을 지나고 있는 경우 (통과 중 센서 값이 20 미만인 상태)
                    else:
                         self.get_logger().info(f"Passing alongside... L: {left_ultra}, R: {right_ultra}, LB: {left_back_ultra}, RB: {right_back_ultra}")

            # [상태 3] 다중 차량 추종 (Time_Gap) + (Straight, 2)의 경우
            elif self.driving_state == "Time_Gap":
                # 조건 1: 차량이 2대 미만이면 즉시 중앙 주행으로 복귀 (상태 탈출)
                if num_detected_vehicles < 2:
                    self.driving_state = "CENTER_DRIVING"
                    current_override_cmd = 'reset'
                    self.get_logger().warn("STATE CHANGE: Time_Gap -> CENTER_DRIVING (Vehicles < 2).")
                
                # 조건 2: 2대 이상이면 Time_Gap 상태 유지
                else:
                    # Time_Gap 상태에서는 항상 중앙 차선을 유지합니다.
                    current_override_cmd = 'reset'
                    
                    # 거리 측정이 가능한 차량이 있다면, 그중 가장 가까운 차를 추종 타겟으로 삼습니다.
                    if processed_obstacles:
                        target_obstacle_for_state_machine = min(processed_obstacles, key=lambda obs: obs['distance'])
                        self.get_logger().info(f"Time_Gap: Following closest vehicle at {target_obstacle_for_state_machine['distance']:.1f}m.")
                    else:
                        self.get_logger().info("Time_Gap: 2+ vehicles detected, but no distance info. Driving straight.")

        # =======================> 확실한 추월 금지 로직 끝 <=========================

        if current_override_cmd != self.last_override_cmd:
            cmd_msg = String(data=current_override_cmd)
            self.override_pub.publish(cmd_msg)
            self.get_logger().warn(f"💡 Lane command sent: '{current_override_cmd}'")
            self.last_override_cmd = current_override_cmd
            
        if is_obstacle_emergency or is_pixel_emergency:
            obstacle_msg.distance_m = 0.0
            if is_obstacle_emergency:
                obstacle_msg.position = emergency_obstacle['position_raw']
            else:
                obstacle_msg.position = "pixel_danger"
            if self.driving_state != "E_STOP":
                 self.get_logger().error("STATE CHANGE: ANY -> E_STOP")
                 self.driving_state = "E_STOP"
        else:
            if self.driving_state == "E_STOP":
                self.driving_state = "CENTER_DRIVING"
                self.get_logger().warn("STATE CHANGE: E_STOP -> CENTER_DRIVING (Obstacle Cleared).")
            if target_obstacle_for_decision:
                obstacle_msg.distance_m = float(target_obstacle_for_decision['distance'])
                obstacle_msg.position = target_obstacle_for_decision['position_final']
                tx, ty = target_obstacle_for_decision['xmin'], target_obstacle_for_decision['ymin']
                cv2.putText(image, f"FINAL TARGET: {final_position_str}", (tx, ty - 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
            else:
                obstacle_msg.distance_m = 999.0
                obstacle_msg.position = "none"

        obstacle_msg.vehicle_count = len(detections)
        self.obstacle_state_pub.publish(obstacle_msg)

        # ================================ 요구사항 추가 부분 시작 ================================
        h, w, _ = image.shape
        closest_lane = 'None'
        farthest_lane = 'None'
        closest_vehicle_ratio = 'N/A'
        if len(processed_obstacles) >= 2:
            sorted_obstacles_by_dist = sorted(processed_obstacles, key=lambda x: x['distance'])
            farthest_lane = sorted_obstacles_by_dist[-1]['position_raw']
        if target_obstacle_for_decision:
            closest_lane = final_position_str # 최종 결정 값(smoothed)을 사용하도록 수정
            closest_vehicle_ratio = target_obstacle_for_decision.get('area_ratio_str', 'N/A')
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        thickness = 2
        line_type = cv2.LINE_AA
        cv2.putText(image, f"closest vehicle's lane : {closest_lane}", (w - 290, 30), font, font_scale, font_color, thickness, line_type)
        cv2.putText(image, f"farthest vehicle's lane : {farthest_lane}", (w - 290, 60), font, font_scale, font_color, thickness, line_type)
        cv2.putText(image, f"Clip Ratio (Closest): {closest_vehicle_ratio}", (w - 330, 90), font, font_scale, font_color, thickness, line_type)
        # ================================ 요구사항 추가 부분 끝 ==================================

        return image

    def synchronized_callback(self, image_msg, curve_msg, yolo_msg, scan_msg, ultra_msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
        
        left_ultra = 100
        left_back_ultra = 999
        right_ultra = 100
        right_back_ultra = 999
        if len(ultra_msg.data) > 0: left_ultra = ultra_msg.data[0]
        if len(ultra_msg.data) > 7: left_back_ultra = ultra_msg.data[7]      
        if len(ultra_msg.data) > 5: right_back_ultra = ultra_msg.data[5]          
        if len(ultra_msg.data) > 4: right_ultra = ultra_msg.data[4]
        else: self.get_logger().warn(f"Ultrasonic data only has {len(ultra_msg.data)} elements. Can't access index 4.")

        ranges = np.array(scan_msg.ranges)
        angles = scan_msg.angle_min + np.arange(len(scan_msg.ranges)) * scan_msg.angle_increment
        mask = (ranges > scan_msg.range_min) & (ranges < 5.0)
        x = ranges[mask] * np.cos(angles[mask])
        y = ranges[mask] * np.sin(angles[mask])
        points_3d = np.stack([x, y, np.zeros_like(x)], axis=1)
        
        img_pts = np.array([])
        if points_3d.shape[0] > 0:
            cam_pts = (self.R @ points_3d.T + self.t).T
            img_pts_raw, _ = cv2.projectPoints(cam_pts, np.zeros(3), np.zeros(3), self.mtx, None)
            img_pts = img_pts_raw.squeeze(axis=1)
                
        if curve_msg.points:
            # 주행 경로를 선으로 그립니다 (기존 코드)
            curve_pts = np.array([[p.x, p.y] for p in curve_msg.points], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [curve_pts], isClosed=False, color=(255, 0, 255), thickness=2)
            
            # 주행 경로를 구성하는 실제 점들을 원으로 그립니다
            for p in curve_msg.points:
                cv2.circle(frame, (int(p.x), int(p.y)), 3, (0, 255, 255), -1) # 노란색(-1: 채워진 원)

        if curve_msg.state:
            cv2.putText(frame, f"Lane State: {curve_msg.state}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        vehicle_count = sum(1 for det in yolo_msg.detections if det.class_name == 'obstacle_vehicle')
        for det in yolo_msg.detections:
            if det.class_name == 'obstacle_vehicle':
                x1, y1, x2, y2 = map(int, [det.xmin, det.ymin, det.xmax, det.ymax])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Vehicles Detected: {vehicle_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(frame, f"State: {self.driving_state}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Left Ultra: {left_ultra}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Left Back Ultra: {left_back_ultra}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Right Ultra: {right_ultra}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Right Back Ultra: {right_back_ultra}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        frame = self.project_lidar_on_image(frame, yolo_msg, points_3d, curve_msg,left_ultra,right_ultra,left_back_ultra,right_back_ultra,img_pts)
        cv2.imshow("target_lane_planner : Integrated Visualization", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = TargetLanePlanner()
    try: rclpy.spin(node)
    except KeyboardInterrupt: node.get_logger().info('KeyboardInterrupt detected, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()