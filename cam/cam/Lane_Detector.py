#! /usr/bin/env python
# -*- coding:utf-8 -*-
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from std_msgs.msg import String
import cv2
import numpy as np
from collections import deque
from custom_interfaces.msg import XycarState, Detections, Curve
import message_filters
from std_srvs.srv import Trigger

# ===============================
# 13pixels = 2.5cm (차선 폭)
# 82.5cm (좌우차선 중앙을 끝점으로)
# 41.25cm = 214.5pixels
# ===============================
clicked_points = []


class Lane_Detector(Node):

    def __init__(self):
        super().__init__('lane_detector')

        # --- Subscriber 및 Publisher 설정 ---
        self.trigger_sub = self.create_subscription(String,'/lane_override_cmd', self.trigger_callback, 10)
        self.img_sub = message_filters.Subscriber(self, Image, '/image_raw')
        self.detections_sub = message_filters.Subscriber(self, Detections, '/yolo_detections')
        self.center_curve_sub = message_filters.Subscriber(self, Curve, '/center_curve')
        
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.img_sub, self.detections_sub, self.center_curve_sub],
            queue_size=10, slop=0.1
        )
        self.time_synchronizer.registerCallback(self.sync_callback)
        self.xycar_state_pub = self.create_publisher(XycarState, '/xycar_state', 10)

        # --- 서비스 서버 생성 및 상태 플래그 초기화 ---
        self.is_active = False
        self.start_service = self.create_service(
            Trigger,
            '/start_track_driving',
            self.start_service_callback
        )

        self.bridge = CvBridge()
        self.get_logger().info('🛣️  Lane_Detector Node Started Successfully❗')
        self.get_logger().info('Waiting for service call on /start_track_driving to begin processing...')

        # 0. 초기값
        self.angle = 0.0
        self.lane = np.array([120.0, 320.0, 520.0])
        self.M = None
        self.bev_curve_points = None
        self.roi_src_poly = None


        # 1. mask_detected_objects
        self.last_seen_map = {}
        self.grid_cell_size = 50
        self.frame_counter = 0

        # 2. apply_canny
        self.canny_low = 50
        self.canny_high = 150

        # 3. detect_lines_hough 
        self.hough_threshold = 25
        self.min_gap = 1
        self.min_length = 10

        # 4. filter_lines_by_angle
        self.angle_tolerance = np.radians(30)
        self.prev_angle = deque([0.0], maxlen=3)

        # 5. extract_lane_candidates_from_clusters
        self.cluster_threshold = 30

        # 6. predict_lane
        self.left_offset = -200
        self.right_offset = 200
        self.angle_correction_gain = 80
        self.min_cos_angle = 0.5
        self.angle_prev_weight = 0.7
        self.angle_new_weight  = 0.3

        # 7. refine_lane_with_candidates_and_prediction
        self.left_to_center_dist = 200
        self.right_to_center_dist = 200
        self.outer_lane_dist = self.left_to_center_dist + self.right_to_center_dist
        self.cluster_match_threshold = 70
        self.lane_update_weight = 0.7
        self.prediction_weight = 0.3
        self.max_lane_delta = 50
        self.center_correction_weight = 0.6
        # <<< 1. 조건부 보정을 위한 임계값 추가
        self.center_agreement_threshold = 80 # 단위: 픽셀, 노란 점과 커브의 오차 임계값. 
        #50이 하한 마지노선 (더 내려가면 커브만 따르는 경향 커짐). 
        # 100이 디폴트. (차선 못 잡는다면 값을 10~20씩 내려봐도 됨.)
        # 값이 내려갈 수록 /center_curve에 맞게 보정이 강해짐. 단점은, 커브에서 노란 점의 진동이 커짐.

        # 8. trigger_callback
        self.override_target_lane = None

    def start_service_callback(self, request, response):
        if not self.is_active:
            self.is_active = True
            self.get_logger().info('✅ Service called. Starting lane detection main logic!')
            response.success = True
            response.message = 'Lane detection started.'
        else:
            self.get_logger().warn('⚠️ Main logic is already running.')
            response.success = False
            response.message = 'Already active.'
        return response

    def get_birds_eye_view(self, image):
        height, width = image.shape[:2]
        output_width = 640
        output_height = 120
        src = np.float32([
            [width * 0.16, height * 0.64],
            [width * 0.84, height * 0.64],
            [width * 1.00, height * 0.72],
            [width * 0.00, height * 0.72]
        ])
        dst = np.float32([
            [output_width * 0.1, 0],
            [output_width * 0.9, 0],
            [output_width * 0.9, output_height],
            [output_width * 0.1, output_height]
        ])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.roi_src_poly = src 
        bev = cv2.warpPerspective(image, self.M, (output_width, output_height))
        self.bev_img_w = output_width
        self.bev_img_h = output_height
        self.bev_img_mid = output_height // 2
        return bev, src

    def transform_curve_to_bev(self, curve_msg):
        if self.M is None or curve_msg is None or not curve_msg.points or self.roi_src_poly is None:
            self.bev_curve_points = None
            return
        all_points = curve_msg.points
        points_inside_roi = [p for p in all_points if cv2.pointPolygonTest(self.roi_src_poly, (p.x, p.y), False) >= 0]
        if not points_inside_roi:
            self.bev_curve_points = None
            return
        original_points = np.array([[[p.x, p.y] for p in points_inside_roi]], dtype=np.float32)
        bev_points = cv2.perspectiveTransform(original_points, self.M)
        if bev_points is not None:
            self.bev_curve_points = bev_points[0]
        else:
            self.bev_curve_points = None
            
    # <<< 2. 조건부 보정 로직으로 수정
    def force_correct_lane_with_curve(self):
        """
        1단계 추정치와 중앙선(YOLO) 정보의 차이가 클 때만 강제 보정을 수행한다.
        """
        if self.bev_curve_points is None or len(self.bev_curve_points) == 0:
            return

        curve_center_x = np.mean(self.bev_curve_points[:, 0])
        
        if not (0 <= curve_center_x < self.bev_img_w):
            return
        
        # 1단계 추정치(self.lane[1])와 YOLO 중앙선(curve_center_x)의 거리 차이 계산
        delta = abs(self.lane[1] - curve_center_x)

        # 거리 차이가 설정된 임계값을 초과할 경우에만 보정 수행
        if delta > self.center_agreement_threshold:
            self.get_logger().warn(f'Large deviation detected ({delta:.1f}px). Overriding center lane.')
            
            # 부드러운 업데이트를 위해 현재 중앙차선 위치와 가중 평균
            new_center = (self.center_correction_weight * curve_center_x) + \
                         ((1 - self.center_correction_weight) * self.lane[1])
            
            self.lane[1] = new_center # 중앙 차선 위치 강제 업데이트
            
            # 새로운 중앙 차선을 기준으로 좌/우 차선 위치 재계산
            cos_angle = max(np.cos(self.angle), self.min_cos_angle)
            self.lane[0] = self.lane[1] - self.left_to_center_dist / cos_angle
            self.lane[2] = self.lane[1] + self.right_to_center_dist / cos_angle

    def apply_canny(self, img, show=False):
        if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (7,7), 0)
        img = cv2.Canny(img, self.canny_low, self.canny_high)
        if show: cv2.imshow('canny', img)
        return img

    def mask_detected_objects(self, canny_img, detections_msg, original_frame, show=False):
        self.frame_counter += 1
        modified_canny = canny_img.copy()
        if show:
            viz_img = original_frame.copy()
            final_mask_for_viz = np.zeros_like(modified_canny)
        active_target_bboxes, current_grid_cells = set(), set()
        current_target_detections = [d for d in detections_msg.detections if d.class_name in ["checkerboard", "obstacle_vehicle"]]
        lane_detections = [d for d in detections_msg.detections if d.class_name == "center_lane"]
        for det in current_target_detections:
            bbox = (det.xmin, det.ymin, det.xmax, det.ymax)
            active_target_bboxes.add(bbox)
            center_x, center_y = (det.xmin + det.xmax) // 2, (det.ymin + det.ymax) // 2
            grid_cell = (center_x // self.grid_cell_size, center_y // self.grid_cell_size)
            current_grid_cells.add(grid_cell)
            self.last_seen_map[grid_cell] = {'last_frame': self.frame_counter, 'bbox': bbox}
        for cell, data in list(self.last_seen_map.items()):
            if self.frame_counter - 1 == data['last_frame'] and cell not in current_grid_cells:
                active_target_bboxes.add(data['bbox'])
            if data['last_frame'] < self.frame_counter - 1:
                del self.last_seen_map[cell]
        for t_bbox in list(active_target_bboxes):
            t_xmin, t_ymin, t_xmax, t_ymax = t_bbox
            overlapping_lanes_bboxes = []
            for lane_det in lane_detections:
                l_xmin, l_ymin, l_xmax, l_ymax = lane_det.xmin, lane_det.ymin, lane_det.xmax, lane_det.ymax
                if not (t_xmax < l_xmin or t_xmin > l_xmax or t_ymax < l_ymin or t_ymin > l_ymax):
                    overlapping_lanes_bboxes.append((l_xmin, l_ymin, l_xmax, l_ymax))
            current_mask = np.zeros_like(modified_canny)
            if not overlapping_lanes_bboxes: cv2.rectangle(current_mask, (t_xmin, t_ymin), (t_xmax, t_ymax), 255, -1)
            else:
                cv2.rectangle(current_mask, (t_xmin, t_ymin), (t_xmax, t_ymax), 255, -1)
                for l_bbox in overlapping_lanes_bboxes: cv2.rectangle(current_mask, (l_bbox[0], l_bbox[1]), (l_bbox[2], l_bbox[3]), 0, -1)
            modified_canny[current_mask == 255] = 0
            if show: final_mask_for_viz = cv2.bitwise_or(final_mask_for_viz, current_mask)
        if show:
            viz_img[final_mask_for_viz == 255] = (0, 0, 0)
            contours, _ = cv2.findContours(final_mask_for_viz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(viz_img, contours, -1, (0, 0, 255), 2)
            cv2.imshow("Object Masking Visualization", viz_img)
            cv2.imshow("Masked Canny Image", modified_canny)
        return modified_canny

    def detect_lines_hough(self, img, show=False):
        lines = cv2.HoughLinesP(img, 1, np.pi/180, self.hough_threshold, self.min_gap, self.min_length)
        if show:
            hough_img = np.zeros((img.shape[0], img.shape[1], 3))
            if lines is not None:
                for x1, y1, x2, y2 in lines[:, 0]: cv2.line(hough_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow('hough', hough_img)
        return lines

    def filter_lines_by_angle(self, lines, show=True):
        thetas, positions = [], []
        if show: filter_img = np.zeros((self.bev_img_h, self.bev_img_w, 3))
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                if y1 == y2: continue
                flag = 1 if y1-y2 > 0 else -1
                theta = np.arctan2(flag * (x2-x1), flag * (y1-y2))
                if abs(theta - self.angle) < self.angle_tolerance:
                    position = float((x2-x1)*(self.bev_img_mid-y1))/(y2-y1) + x1
                    thetas.append(theta)
                    positions.append(position)
                    if show: cv2.line(filter_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        self.prev_angle.append(self.angle)
        if thetas:
            new_angle = np.mean(thetas)
            self.angle = self.angle_prev_weight * self.angle + self.angle_new_weight * new_angle
        if show: cv2.imshow('filtered lines by angle', filter_img)
        return positions

    def extract_lane_candidates_from_clusters(self, positions, show=False, base_img=None):
        clusters = []
        for position in positions:
            if 0 <= position < self.bev_img_w:
                for cluster in clusters:
                    if abs(np.mean(cluster) - position) < self.cluster_threshold:
                        cluster.append(position)
                        break
                else: clusters.append([position])
        lane_candidates = [np.mean(cluster) for cluster in clusters]
        if show and base_img is not None:
            cluster_img = cv2.cvtColor(base_img.copy(), cv2.COLOR_GRAY2BGR)
            colors = [tuple(map(int, np.random.randint(0, 255, size=3))) for _ in range(len(clusters))]
            for idx, cluster in enumerate(clusters):
                for x in cluster: cv2.circle(cluster_img, (int(x), self.bev_img_mid), 4, colors[idx], -1)
            cv2.imshow('Clustered Lane Positions', cluster_img)
        return lane_candidates

    def predict_lane(self, show=False, base_img=None):
        denom = max(np.cos(self.angle), self.min_cos_angle)
        predicted_lane = self.lane[1] + np.array([self.left_offset / denom, 0, self.right_offset / denom])
        predicted_lane += (self.angle - np.mean(self.prev_angle)) * self.angle_correction_gain
        if show and base_img is not None:
            img = cv2.cvtColor(base_img.copy(), cv2.COLOR_GRAY2BGR) if len(base_img.shape) == 2 else base_img.copy()
            y = self.bev_img_mid
            cv2.circle(img, (int(predicted_lane[0]), y), 4, (255, 0, 255), 2)
            cv2.circle(img, (int(predicted_lane[1]), y), 4, (128, 128, 128), 2)
            cv2.circle(img, (int(predicted_lane[2]), y), 4, (255, 255, 0), 2)
            cv2.putText(img, 'Pred_L', (int(predicted_lane[0])-20, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1)
            cv2.putText(img, 'Pred_C', (int(predicted_lane[1])-20, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128,128,128), 1)
            cv2.putText(img, 'Pred_R', (int(predicted_lane[2])-20, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
            cv2.imshow('Predicted Lane', img)
        return predicted_lane
        
    def refine_lane_with_candidates_and_prediction(self, lane_candidates, predicted_lane):
            # if self.override_target_lane == 'go_left':
            #     if predicted_lane[0] < 0 or predicted_lane[0] >= self.bev_img_w:
            #         self.get_logger().warn('⚠️ Left lane out of view, switching back to normal mode')
            #         self.override_target_lane = None
            # elif self.override_target_lane == 'go_right':
            #     if predicted_lane[2] < 0 or predicted_lane[2] >= self.bev_img_w:
            #         self.get_logger().warn('⚠️ Right lane out of view, switching back to normal mode')
            #         self.override_target_lane = None

            if self.override_target_lane == 'go_left':
                cos_angle = max(np.cos(self.angle), self.min_cos_angle)
                left_candidates = [lc for lc in lane_candidates if abs(lc - predicted_lane[0]) < self.cluster_match_threshold]
                if left_candidates:
                    best_left = min(left_candidates, key=lambda x: abs(x - predicted_lane[0]))
                    new_lane = np.array([best_left, best_left + self.left_to_center_dist / cos_angle, best_left + self.outer_lane_dist / cos_angle])
                    delta = np.clip(new_lane - self.lane, -self.max_lane_delta, self.max_lane_delta)
                    self.lane += delta
                else: self.lane = predicted_lane
                return
                    
            elif self.override_target_lane == 'go_right':
                cos_angle = max(np.cos(self.angle), self.min_cos_angle)
                right_candidates = [lc for lc in lane_candidates if abs(lc - predicted_lane[2]) < self.cluster_match_threshold]
                if right_candidates:
                    best_right = min(right_candidates, key=lambda x: abs(x - predicted_lane[2]))
                    new_lane = np.array([best_right - self.outer_lane_dist / cos_angle, best_right - self.right_to_center_dist / cos_angle, best_right])
                    delta = np.clip(new_lane - self.lane, -self.max_lane_delta, self.max_lane_delta)
                    self.lane += delta
                else: self.lane = predicted_lane
                return

            if not lane_candidates: self.lane = predicted_lane; return
            
            possibles, cos_angle = [], max(np.cos(self.angle), self.min_cos_angle)
            for lc in lane_candidates:
                idx = np.argmin(abs(self.lane - lc))
                if idx == 0:
                    est = [lc, lc + self.left_to_center_dist/cos_angle, lc + self.outer_lane_dist/cos_angle]
                    lc2_cands = [c for c in lane_candidates if abs(c - est[1])<self.cluster_match_threshold] or [est[1]]
                    lc3_cands = [c for c in lane_candidates if abs(c - est[2])<self.cluster_match_threshold] or [est[2]]
                    for lc2 in lc2_cands: possibles.extend([[lc, lc2, lc3] for lc3 in lc3_cands])
                elif idx == 1:
                    est = [lc - self.left_to_center_dist/cos_angle, lc, lc + self.right_to_center_dist/cos_angle]
                    lc1_cands = [c for c in lane_candidates if abs(c - est[0])<self.cluster_match_threshold] or [est[0]]
                    lc3_cands = [c for c in lane_candidates if abs(c - est[2])<self.cluster_match_threshold] or [est[2]]
                    for lc1 in lc1_cands: possibles.extend([[lc1, lc, lc3] for lc3 in lc3_cands])
                else:
                    est = [lc - self.outer_lane_dist/cos_angle, lc - self.right_to_center_dist/cos_angle, lc]
                    lc1_cands = [c for c in lane_candidates if abs(c - est[0])<self.cluster_match_threshold] or [est[0]]
                    lc2_cands = [c for c in lane_candidates if abs(c - est[1])<self.cluster_match_threshold] or [est[1]]
                    for lc1 in lc1_cands: possibles.extend([[lc1, lc2, lc] for lc2 in lc2_cands])

            possibles = np.array(possibles)
            best = possibles[np.argmin(np.sum((possibles - predicted_lane)**2, axis=1))]
            new_lane = self.lane_update_weight * best + self.prediction_weight * predicted_lane
            delta = np.clip(new_lane - self.lane, -self.max_lane_delta, self.max_lane_delta)
            self.lane += delta

    def mark_lane(self, img, lane=None, show=False, curve_points=None):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if lane is None: lane = self.lane
        l1, l2, l3 = lane
        cv2.circle(img, (int(l1), self.bev_img_mid), 3, (255, 0, 0), 5, cv2.FILLED)
        cv2.circle(img, (int(l2), self.bev_img_mid), 3, (0, 255, 255), 5, cv2.FILLED)
        cv2.circle(img, (int(l3), self.bev_img_mid), 3, (0, 0, 255), 5, cv2.FILLED)

        mid_left, mid_right = (l1 + l2) / 2, (l2 + l3) / 2
        cv2.circle(img, (int(mid_left), self.bev_img_mid), 4, (0, 255, 0), 4, cv2.FILLED)
        cv2.putText(img, 'go_L', (int(mid_left)-20, self.bev_img_mid-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.circle(img, (int(mid_right), self.bev_img_mid), 4, (0, 255, 0), 4, cv2.FILLED)
        cv2.putText(img, 'go_R', (int(mid_right)-20, self.bev_img_mid-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if curve_points is not None and len(curve_points) > 0:
            curve_points_int = curve_points.astype(np.int32)
            cv2.polylines(img, [curve_points_int], isClosed=False, color=(255, 0, 255), thickness=2)

        if self.override_target_lane:
            cv2.putText(img, f'Override: {self.override_target_lane}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # if show:
            # cv2.imshow('marked_lane_point', img)

    def visualize_results_on_original_frame(self, original_img, roi_poly, curve_msg, lane_center_bev, curve_center_bev, show=False):
        """
        요청사항을 반영한 새로운 시각화 함수:
        1. 원본 이미지에 ROI 영역(초록색)을 표시합니다.
        2. 원본 이미지에 구독한 중앙선(노란색)을 그립니다.
        3. BEV 공간에서 계산된 차선 검출 중앙값과 YOLO 중앙값 간의 오차를 좌측 상단에 표시합니다.
        """
        if not show:
            return

        display_img = original_img.copy()

        # 1. ROI 영역 표시
        if roi_poly is not None:
            cv2.polylines(display_img, [roi_poly.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)

        # 2. 구독한 curve(YOLO 중앙선) 표시
        if curve_msg is not None and curve_msg.points:
            curve_points = np.array([[[int(p.x), int(p.y)]] for p in curve_msg.points], dtype=np.int32)
            cv2.polylines(display_img, [curve_points], isClosed=False, color=(255, 255, 0), thickness=2)

        # 3. BEV 공간에서의 오차 계산 및 표시
        error_text = "Error: N/A"
        if lane_center_bev is not None and curve_center_bev is not None:
            # "노란 점"(차선 검출 결과)과 "curve"(YOLO 결과)의 거리 오차
            error = abs(lane_center_bev - curve_center_bev)
            error_text = f"Error: {error:.2f} px"
        
        cv2.putText(display_img, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # 최종 결과 창 표시
        # cv2.imshow('Result on Original Frame', display_img)

    def show_roi_region(self, img, src, show=False):
        if show:
            cv2.polylines(img, [src.astype(int)], True, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow('roi region on the undistorted frame',img)

    def sync_callback(self, img_msg, detections_msg, curve_msg):
        if not self.is_active: return
            
        frame = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        canny = self.apply_canny(frame, show=False)
        canny = self.mask_detected_objects(canny, detections_msg, frame, show=False)
        
        bev, src = self.get_birds_eye_view(canny)
        self.transform_curve_to_bev(curve_msg)
        
        # self.show_roi_region(frame, src, show=False) # 새로운 함수로 대체됨
        lines = self.detect_lines_hough(bev, show=False)
        positions = self.filter_lines_by_angle(lines, show=False)
        lane_candidates = self.extract_lane_candidates_from_clusters(positions, show=False, base_img=bev)

        predicted_lane = self.predict_lane(show=False, base_img=bev)
        
        self.refine_lane_with_candidates_and_prediction(lane_candidates, predicted_lane)
        self.force_correct_lane_with_curve()
        
        self.mark_lane(bev, show=True, curve_points=self.bev_curve_points)
        
        # --- [변경] 새로운 시각화 함수 호출 ---
        # BEV 공간에서 YOLO 중앙선의 x좌표 평균 계산
        curve_center_bev_x = None
        if self.bev_curve_points is not None and len(self.bev_curve_points) > 0:
            curve_center_bev_x = np.mean(self.bev_curve_points[:, 0])

        # 새로운 시각화 함수 호출 (show=True로 설정하여 활성화)
        self.visualize_results_on_original_frame(
            original_img=frame,
            roi_poly=self.roi_src_poly,
            curve_msg=curve_msg,
            lane_center_bev=self.lane[1],
            curve_center_bev=curve_center_bev_x,
            show=True
        )
        # ------------------------------------

        state_msg = XycarState()
        if self.override_target_lane == 'go_right':
            state_msg.drive_mode = 'right'
            target_x = (self.lane[1] + self.lane[2]) / 2
        elif self.override_target_lane == 'go_left':
            state_msg.drive_mode = 'left'
            target_x = (self.lane[0] + self.lane[1]) / 2
        else:
            state_msg.drive_mode = 'center'
            target_x = self.lane[1]
        state_msg.target_point.x = float(target_x)
        state_msg.target_point.y = float(self.bev_img_mid)
        state_msg.target_point.z = self.angle
        self.xycar_state_pub.publish(state_msg)
        cv2.waitKey(1)
        
    def trigger_callback(self, msg):
        command = msg.data.strip().lower()
        if command == 'go_right': self.override_target_lane = 'go_right'
        elif command == 'go_left': self.override_target_lane = 'go_left'
        elif command == 'reset': self.override_target_lane = None
        else: self.get_logger().warn(f'⚠️ [Command Received] unknown: {command}')
    
def main(args=None):
    rclpy.init(args=args)
    node = Lane_Detector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()