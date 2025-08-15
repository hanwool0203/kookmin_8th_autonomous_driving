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

# ===============================
# 13pixels = 2.5cm (차선 폭)
# 82.5cm (좌우차선 중앙을 끝점으로)
# 41.25cm = 214.5pixels
# ===============================

# =============================================================================
# 오른쪽 차선(중앙 + 오른쪽 중간)으로 주행
# ros2 topic pub --once /lane_override_cmd std_msgs/String "data: 'go_right'"

# 왼쪽 차선(왼쪽 + 중앙 중간)으로 주행
# ros2 topic pub --once /lane_override_cmd std_msgs/String "data: 'go_left'"

# 다시 중앙차선으로 복귀
# ros2 topic pub --once /lane_override_cmd std_msgs/String "data: 'reset'"
# =============================================================================

clicked_points = []

### ===== 마우스 클릭으로 두 점을 선택해 x축 거리(px)를 계산하고 이미지에 표시한다 =====
def click_and_measure(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f'Point {len(clicked_points)}: ({x}, {y})')
        if len(clicked_points) == 2:
            pt1, pt2 = clicked_points
            dist = abs(pt1[0] - pt2[0])
            print(f'X-distance between points: {dist:.2f} pixels')
            marked_img = param.copy()
            cv2.line(marked_img, pt1, pt2, (255, 255, 0), 2)
            mid_pt = ((pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2)
            cv2.putText(marked_img, f"X distance{dist:.1f}px", mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("marked", marked_img)
            clicked_points = []

class Lane_Detector(Node):

    def __init__(self):
        super().__init__('lane_detector')

        self.img_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.trigger_sub = self.create_subscription(String,'/lane_override_cmd', self.trigger_callback, 10)
        self.lane_point_pub = self.create_publisher(Point, '/lane_point', 10)

        # 🔧 추가: 상태/이벤트 퍼블리셔
        self.mode_pub  = self.create_publisher(String, '/lane_mode', 10)
        self.current_mode = 'center'   # 'center' | 'left' | 'right'

        self.bridge = CvBridge()
        self.get_logger().info('🛣️  Lane_Detector Node Started Successfully❗')

        # 0. 초기값
        self.angle = 0.0
        self.lane = np.array([120.0, 320.0, 520.0])  # 초기 왼쪽, 중앙, 오른쪽 차선의 x좌표

        # 2. apply_canny
        self.canny_low = 50
        self.canny_high = 150

        # 3. detect_lines_hough 
        self.hough_threshold = 25
        self.min_gap = 1
        self.min_length = 10

        # 4. filter_lines_by_angle
        self.angle_tolerance = np.radians(30)       # y=60px에서 현재 차선 방향과의 허용 각도 차이 (degree)
        self.prev_angle = deque([0.0], maxlen=3)    # (6 에서도 사용됨)

        # 5. extract_lane_candidates_from_clusters
        self.cluster_threshold = 30                 # 위치값들을 클러스터로 묶을 때, 평균값과의 최대 허용 거리(임계값)

        # 6. predict_lane
        self.left_offset = -200
        self.right_offset = 200
        self.angle_correction_gain = 80             # 예측 차선 위치 보정 시 각도 변화량에 곱해질 보정 계수 
        self.min_cos_angle = 0.5                    # (7 에서도 사용됨) 차선 각도가 너무 기울어질 경우를 방지하기 위한 최소 cos(각도) 값
        self.angle_prev_weight = 0.7                # 이전 각도에 대한 가중치
        self.angle_new_weight  = 0.3                # 현재 프레임 각도에 대한 가중치

        # 7. refine_lane_with_candidates_and_prediction
        self.left_to_center_dist = 200
        self.right_to_center_dist = 200
        self.outer_lane_dist = self.left_to_center_dist + self.right_to_center_dist
        self.cluster_match_threshold = 70           ### 예상된 차선 위치와 실제 후보 간의 허용 거리
        self.lane_update_weight = 0.7               ### 최종 차선 위치 계산 시 현재 관측값에 대한 반영 비율
        self.prediction_weight = 0.3                ### 최종 차선 위치 계산 시 예측값에 대한 반영 비율
        self.max_lane_delta = 30                    # 한 프레임에서 허용되는 최대 차선 변화량 [픽셀]

        # 8. trigger_callback
        self.override_target_lane = None            # 'go_right', 'go_left' 등


    ### ===== 원근 변환을 통해 영상을 탑다운 시점의 Bird's Eye View로 변환한다 =====
    def get_birds_eye_view(self, image):
        height, width = image.shape[:2]
        output_width = 640
        output_height = 120
        src = np.float32([
            [width * 0.16, height * 0.64],      # TOP-LEFT
            [width * 0.84, height * 0.64],      # TOP-RIGHT
            [width * 1.00, height * 0.72],      # BOT-RIGHT
            [width * 0.00, height * 0.72]       # BOT-LEFT
        ])
        dst = np.float32([
            [output_width * 0.1, 0],
            [output_width * 0.9, 0],
            [output_width * 0.9, output_height],
            [output_width * 0.1, output_height]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        bev = cv2.warpPerspective(image, M, (output_width, output_height))
        self.bev_img_w = output_width
        self.bev_img_h = output_height
        self.bev_img_mid = output_height // 2
        return bev, src

    ### ===== 가우시안 블러 후 캐니 알고리즘으로 이미지의 에지(윤곽선)를 검출한다 =====
    def apply_canny(self, img, show=False):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (7,7), 0)
        img = cv2.Canny(img, self.canny_low, self.canny_high)
        if show:
            cv2.imshow('canny', img)
        return img

    ### ===== 확률적 허프 변환을 이용해 이미지에서 직선(선분)들을 검출한다 =====
    def detect_lines_hough(self, img, show=False):
        lines = cv2.HoughLinesP(img, 1, np.pi/180, self.hough_threshold, self.min_gap, self.min_length)
        if show:
            hough_img = np.zeros((img.shape[0], img.shape[1], 3))
            if lines is not None:
                for x1, y1, x2, y2 in lines[:, 0]:
                    cv2.line(hough_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow('hough', hough_img)
        return lines

    ### ===== 현재 차선 각도와 유사한 직선들만 필터링하여 차선 후보 위치(x좌표)를 추출한다 =====
    def filter_lines_by_angle(self, lines, show=True):
        thetas, positions = [], []
        if show:
            filter_img = np.zeros((self.bev_img_h, self.bev_img_w, 3))
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                if y1 == y2:
                    continue
                flag = 1 if y1-y2 > 0 else -1
                theta = np.arctan2(flag * (x2-x1), flag * (y1-y2))      # 허프 변환으로 검출된 두 끝점으로부터 계산
                if abs(theta - self.angle) < self.angle_tolerance:      # 현재 선분의 방향(theta)과 과거 차선 방향(self.angle)의 차이
                    position = float((x2-x1)*(self.bev_img_mid-y1))/(y2-y1) + x1    # 60px에서 각 직선의 위치 x (단일)
                    thetas.append(theta)
                    positions.append(position)                                      # 60px에서 각 직선의 위치 x (리스트)
                    if show:
                        cv2.line(filter_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        self.prev_angle.append(self.angle)
        if thetas:
            new_angle = np.mean(thetas)                                 # 이전 프레임에서 필터링된 모든 선의 각도 리스트 theta의 평균
            self.angle = self.angle_prev_weight * self.angle + self.angle_new_weight * new_angle
        if show:
            cv2.imshow('filtered lines by angle', filter_img)
        return positions

    ### ===== 위치값들을 클러스터링하여 평균값 기준의 차선 후보 좌표들을 추출한다 =====
    def extract_lane_candidates_from_clusters(self, positions, show=False, base_img=None):
        clusters = []                               # 여러 개의 cluster를 담은 전체 리스트 (2차원 리스트)
        for position in positions:
            if 0 <= position < self.bev_img_w:
                for cluster in clusters:            # 유사한 x좌표들을 모은 하나의 그룹 (1차원 리스트)
                    cluster_mean = np.mean(cluster)
                    if abs(cluster_mean - position) < self.cluster_threshold:
                        cluster.append(position)
                        break
                else:
                    clusters.append([position])
        lane_candidates = [np.mean(cluster) for cluster in clusters]

        if show and base_img is not None:
            cluster_img = cv2.cvtColor(base_img.copy(), cv2.COLOR_GRAY2BGR)
            np.random.seed(42)
            colors = [tuple(map(int, np.random.randint(0, 255, size=3))) for _ in range(len(clusters))]
            for idx, cluster in enumerate(clusters):
                color = colors[idx]
                for x in cluster:
                    cv2.circle(cluster_img, (int(x), self.bev_img_mid), 4, color, -1)
            cv2.imshow('Clustered Lane Positions', cluster_img)
        return lane_candidates

    ### ===== 이전 차선 위치와 각도를 기반으로 현재 프레임의 차선 위치를 예측한다 =====
    def predict_lane(self, show=False, base_img=None):
        denom = max(np.cos(self.angle), self.min_cos_angle)
        predicted_lane = self.lane[1] + np.array([
            self.left_offset / denom,
            0,
            self.right_offset / denom
        ])
        predicted_lane += (self.angle - np.mean(self.prev_angle)) * self.angle_correction_gain

        if show and base_img is not None:
            img = cv2.cvtColor(base_img.copy(), cv2.COLOR_GRAY2BGR) if len(base_img.shape) == 2 else base_img.copy()
            y = self.bev_img_mid
            cv2.circle(img, (int(predicted_lane[0]), y), 4, (255, 0, 255), 2)       # 좌측 예측 (보라)
            cv2.circle(img, (int(predicted_lane[1]), y), 4, (128, 128, 128), 2)     # 중앙 예측 (회색)
            cv2.circle(img, (int(predicted_lane[2]), y), 4, (255, 255, 0), 2)       # 우측 예측 (노랑)
            cv2.putText(img, 'Pred_L', (int(predicted_lane[0]) - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            cv2.putText(img, 'Pred_C', (int(predicted_lane[1]) - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            cv2.putText(img, 'Pred_R', (int(predicted_lane[2]) - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.imshow('Predicted Lane', img)
        return predicted_lane
        
    ### ===== 예측된 차선과 후보 차선을 비교하여 가장 유사한 조합을 선택하고 최종 차선 위치를 보정한다 =====
    def refine_lane_with_candidates_and_prediction(self, lane_candidates, predicted_lane):
        if not lane_candidates:
            self.lane = predicted_lane
            return
        possibles = []
        cos_angle = max(np.cos(self.angle), self.min_cos_angle)
        for lc in lane_candidates:
            idx = np.argmin(abs(self.lane - lc))
            if idx == 0:
                estimated_lane = [
                    lc,
                    lc + self.left_to_center_dist / cos_angle,
                    lc + self.outer_lane_dist / cos_angle
                ]
                lc2_candidate = [c for c in lane_candidates if abs(c - estimated_lane[1]) < self.cluster_match_threshold]
                lc3_candidate = [c for c in lane_candidates if abs(c - estimated_lane[2]) < self.cluster_match_threshold]
                if not lc2_candidate:
                    lc2_candidate.append(estimated_lane[1])
                if not lc3_candidate:
                    lc3_candidate.append(estimated_lane[2])
                for lc2 in lc2_candidate:
                    for lc3 in lc3_candidate:
                        possibles.append([lc, lc2, lc3])
            elif idx == 1:
                estimated_lane = [
                    lc - self.left_to_center_dist / cos_angle,
                    lc,
                    lc + self.right_to_center_dist / cos_angle
                ]
                lc1_candidate = [c for c in lane_candidates if abs(c - estimated_lane[0]) < self.cluster_match_threshold]
                lc3_candidate = [c for c in lane_candidates if abs(c - estimated_lane[2]) < self.cluster_match_threshold]
                if not lc1_candidate:
                    lc1_candidate.append(estimated_lane[0])
                if not lc3_candidate:
                    lc3_candidate.append(estimated_lane[2])
                for lc1 in lc1_candidate:
                    for lc3 in lc3_candidate:
                        possibles.append([lc1, lc, lc3])
            else:
                estimated_lane = [
                    lc - self.outer_lane_dist / cos_angle,
                    lc - self.right_to_center_dist / cos_angle,
                    lc
                ]
                lc1_candidate = [c for c in lane_candidates if abs(c - estimated_lane[0]) < self.cluster_match_threshold]
                lc2_candidate = [c for c in lane_candidates if abs(c - estimated_lane[1]) < self.cluster_match_threshold]
                if not lc1_candidate:
                    lc1_candidate.append(estimated_lane[0])
                if not lc2_candidate:
                    lc2_candidate.append(estimated_lane[1])
                for lc1 in lc1_candidate:
                    for lc2 in lc2_candidate:
                        possibles.append([lc1, lc2, lc])
        
        possibles = np.array(possibles)
        error = np.sum((possibles - predicted_lane) ** 2, axis=1)
        best = possibles[np.argmin(error)]
        new_lane = self.lane_update_weight * best + self.prediction_weight * predicted_lane

        delta = new_lane - self.lane
        delta_clipped = np.clip(delta, -self.max_lane_delta, self.max_lane_delta)
        self.lane = self.lane + delta_clipped

    ### ===== 최종 결정된 차선 좌표들과 그 중앙값들을 이미지에 시각화하여 표시한다 =====
    def mark_lane(self, img, lane=None, show=False):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if lane is None:
            lane = self.lane
        l1, l2, l3 = lane
        cv2.circle(img, (int(l1), self.bev_img_mid), 3, (255, 0, 0), 5, cv2.FILLED)     # BLUE
        cv2.circle(img, (int(l2), self.bev_img_mid), 3, (0, 255, 255), 5, cv2.FILLED)   # YELLOW
        cv2.circle(img, (int(l3), self.bev_img_mid), 3, (0, 0, 255), 5, cv2.FILLED)     # RED

        mid_left = (l1 + l2) / 2
        mid_right = (l2 + l3) / 2
        cv2.circle(img, (int(mid_left), self.bev_img_mid), 4, (0, 255, 0), 4, cv2.FILLED)   # GREEN
        cv2.putText(img, 'go_L', (int(mid_left)-20, self.bev_img_mid - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.circle(img, (int(mid_right), self.bev_img_mid), 4, (0, 255, 0), 4, cv2.FILLED)  # GREEN
        cv2.putText(img, 'go_R', (int(mid_right)-20, self.bev_img_mid - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if show:
            cv2.imshow('marked_lane_point', img)
            cv2.setMouseCallback('marked_lane_point', click_and_measure, img.copy())

    ### ===== 원본 이미지에 BEV 변환 영역을 폴리라인으로 표시하여 시각화한다 =====
    def show_roi_region(self, img, src, show=False):
        if show:
            COLOR = (0,255,0)
            THICKNESS = 2
            cv2.polylines(img, [src.astype(int)], True, COLOR, THICKNESS, cv2.LINE_AA)
            cv2.imshow('roi region on the undistorted frame',img)
        else :
            pass

    ### ===== 수신된 이미지에서 차선을 인식하고, 최종 중앙 차선 위치와 각도를 계산하여 퍼블리시한다 =====
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        canny = self.apply_canny(frame, show=False)
        bev, src = self.get_birds_eye_view(canny)
        self.show_roi_region(frame, src, show=True)
        lines = self.detect_lines_hough(bev, show=False)
        positions = self.filter_lines_by_angle(lines, show=False)
        lane_candidates = self.extract_lane_candidates_from_clusters(positions,show=False,base_img=bev)

        predicted_lane = self.predict_lane(show=False, base_img=bev)
        self.refine_lane_with_candidates_and_prediction(lane_candidates, predicted_lane)
        self.mark_lane(bev, show=True)

        lane_msg = Point()

        if self.override_target_lane == 'go_right':
            mode_str = 'right'
            lane_msg.x = (self.lane[1] + self.lane[2]) / 2
        elif self.override_target_lane == 'go_left':
            mode_str = 'left'
            lane_msg.x = (self.lane[0] + self.lane[1]) / 2
        else:
            mode_str = 'center'
            lane_msg.x = self.lane[1]

        lane_msg.y = float(self.bev_img_mid)
        lane_msg.z = self.angle

        self.lane_point_pub.publish(lane_msg)
        self.mode_pub.publish(String(data=mode_str))

        # 현재 모드가 override 명령과 다를 경우에만 발행하여 불필요한 메시지 줄임
        if mode_str != self.current_mode:
            self.current_mode = mode_str
            self.mode_pub.publish(String(data=self.current_mode))

        cv2.waitKey(1)
        

    ### ===== 외부 명령(go_right/go_left)에 따라 차선 선택을 강제로 지정하거나 초기화한다 =====
    def trigger_callback(self, msg):  # std_msgs/String
        command = msg.data.strip().lower()

        # 1) override_target_lane 먼저 갱신
        if command == 'go_right':
            self.override_target_lane = 'go_right'
        elif command == 'go_left':
            self.override_target_lane = 'go_left'
        elif command == 'reset':
            self.override_target_lane = None
        else:
            self.get_logger().warn(f'⚠️ [Command Received] unknown: {command}')
            return

        # if self.override_target_lane == 'go_right':
        #     new_mode = 'right'
        # elif self.override_target_lane == 'go_left':
        #     new_mode = 'left'
        # else:
        #     new_mode = 'center'

        # if new_mode != self.current_mode:
        #     self.current_mode = new_mode


        
    
def main(args=None):
    rclpy.init(args=args)
    node = Lane_Detector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
