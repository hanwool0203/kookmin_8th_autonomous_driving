#! /usr/bin/env python
# -*- coding:utf-8 -*-
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import subprocess
import time
import threading
import message_filters

# [중요] 사용자 정의 메시지를 임포트합니다.
# 'your_package_name'을 Detections 메시지가 정의된 실제 패키지 이름으로 변경해야 합니다.
# 예: from vision_msgs.msg import Detections
from cam_interfaces.msg import Detections


class CheckerBoard_Detector(Node):

    def __init__(self):
        super().__init__('checkerboard_detector')

        # --- 노드 제어 설정 (변경 없음) ---
        self.nodes_to_kill = [
            "pure_pursuit_node",
            "path_planning_node",
            "preprocessing_node"
        ]
        self.nodes_to_run = [
            {'package': 'cam', 'executable': 'integrated_stanley_controller'},
            {'package': 'cam', 'executable': 'lane_detector'},
            {'package': 'lidar2cam_projector', 'executable': 'lidar2cam_projector'} ,
            {'package': 'lidar2cam_projector', 'executable': 'yolo_lidar_fusion_node'},
            {'package': 'objacc_controller', 'executable': 'objacc_controller'}
        ]
        self.is_shutting_down = False # 종료 시퀀스가 중복 실행되는 것을 방지하기 위한 플래그
        # ------------------------------------

        # --- [수정] 시각화 및 인식 임계값 파라미터 선언 ---
        self.declare_parameter('enable_visualization', True)
        self.declare_parameter('show_class_name', False)
        self.declare_parameter('show_confidence', False)
        # 바운딩 박스 면적 임계값 파라미터 추가
        self.declare_parameter('bbox_area_threshold', 8000) # 기본값 15000

        # 파라미터 값 읽어오기
        self.enable_visualization = self.get_parameter('enable_visualization').get_parameter_value().bool_value
        self.show_class_name = self.get_parameter('show_class_name').get_parameter_value().bool_value
        self.show_confidence = self.get_parameter('show_confidence').get_parameter_value().bool_value
        self.bbox_area_threshold = self.get_parameter('bbox_area_threshold').get_parameter_value().integer_value

        # --- 구독자 설정 ---
        # --- 구독 동기화 ---
        self.detection_sub = message_filters.Subscriber(self, Detections, '/yolo_detections')

        self.img_sub = message_filters.Subscriber(self, Image, '/image_raw')

        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.detection_sub, self.img_sub],
            queue_size=10,
            slop=0.1
        )
        self.time_synchronizer.registerCallback(self.synchronized_callback)
        # -----------------

        self.bridge = CvBridge()
        self.get_logger().info('Final CheckerBoard_Detector Node Started Successfully❗')
        self.get_logger().info(f"Visualization: {'Enabled' if self.enable_visualization else 'Disabled'}, ClassName: {self.show_class_name}, Confidence: {self.show_confidence}")
        self.get_logger().info(f"Checkerboard detection area threshold: {self.bbox_area_threshold} pixels")


        # --- 시각화 및 데이터 공유 설정 ---
        self.result_win_name = 'Detection Result'
        if self.enable_visualization:
            cv2.namedWindow(self.result_win_name)

        self.latest_detections = []
        self.lock = threading.Lock()

    # ------------------ 콜백함수 동기화 ------------------------------
    def synchronized_callback(self, detections_msg, image_msg):
        #self.get_logger().info('<<<<< SYNCHRONIZED CALLBACK IS RUNNING! >>>>>') # 콜백 호출 디버깅
        self.detections_callback(detections_msg)
        self.image_callback(image_msg)
    # --------------------------------------------------------------

    def shutdown_sequence(self):
        # 이 함수의 내용은 변경되지 않았습니다.
        if self.is_shutting_down:
            return
        self.is_shutting_down = True

        self.get_logger().info("--- Checkerboard Detected at sufficient proximity! Initiating node transition sequence. ---")

        # 1. 종료할 노드들을 개별적으로 종료
        for node_name in self.nodes_to_kill:
            try:
                self.get_logger().info(f"Attempting to terminate node: {node_name}")
                result = subprocess.run(f"pkill -f {node_name}", shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    self.get_logger().info(f"Node '{node_name}' terminated successfully.")
                else:
                    self.get_logger().warn(f"Node '{node_name}' was not running or already terminated.")
            except Exception as e:
                self.get_logger().error(f"An unexpected error occurred while terminating node '{node_name}': {e}")

        # 2. 실행할 노드들을 리스트 기반으로 실행
        self.get_logger().info("--- Launching new nodes ---")
        for node_info in self.nodes_to_run:
            package = node_info['package']
            executable = node_info['executable']
            try:
                self.get_logger().info(f"Launching node '{executable}' from package '{package}'...")
                command = f"ros2 run {package} {executable}"
                subprocess.Popen(command, shell=True)
                self.get_logger().info(f"Node '{executable}' launched successfully.")
            except Exception as e:
                self.get_logger().error(f"Failed to launch node '{executable}'. Error: {e}")

        # 3. 자기 자신 노드 종료
        self.get_logger().info("Shutting down CheckerBoard_Detector node.")
        if self.enable_visualization:
            cv2.destroyAllWindows()
        self.destroy_node()

    # --- [수정] Detections 메시지를 처리하는 콜백 함수 ---
    def detections_callback(self, msg):
        """
        Detections 메시지를 받아 객체 정보를 저장하고,
        체커보드가 임계값보다 클 경우 노드 전환 시퀀스를 시작합니다.
        """
        if self.is_shutting_down:
            return

        trigger_shutdown = False
        temp_detections = []

        for detection in msg.detections:
            # 시각화를 위해 모든 감지 정보 저장
            det_info = {
                'class_name': detection.class_name,
                'confidence': detection.confidence,
                'bbox': (
                    int(detection.xmin),
                    int(detection.ymin),
                    int(detection.xmax),
                    int(detection.ymax)
                )
            }
            temp_detections.append(det_info)

            # 체커보드인 경우, 크기(면적)를 확인하여 노드 전환 여부 결정
            if detection.class_name == 'checkerboard':
                xmin, ymin, xmax, ymax = det_info['bbox']
                area = (xmax - xmin) * (ymax - ymin)
                
                # 면적이 임계값 이상일 때만 노드 전환 플래그를 True로 설정
                if area >= self.bbox_area_threshold:
                    self.get_logger().info(f"Checkerboard detected with area: {area} >= threshold: {self.bbox_area_threshold}. Triggering transition.")
                    trigger_shutdown = True
                else:
                    self.get_logger().info(f"Checkerboard detected with area: {area} < threshold: {self.bbox_area_threshold}. Too small, ignoring.")

        # 감지된 객체 리스트 업데이트 (시각화용)
        with self.lock:
            self.latest_detections = temp_detections

        # 노드 전환 플래그가 True일 때만 종료 시퀀스 실행
        if trigger_shutdown:
            self.shutdown_sequence()

    # --- [수정 없음] 이미지를 시각화하는 콜백 함수 ---
    def image_callback(self, msg):
        """
        카메라 이미지를 수신하여 설정에 따라 검출 결과를 시각화합니다.
        """
        if self.is_shutting_down or not self.enable_visualization:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        with self.lock:
            local_detections = list(self.latest_detections)

        for det in local_detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            xmin, ymin, xmax, ymax = bbox
            color = (0, 255, 0)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            label_parts = []
            if self.show_class_name:
                label_parts.append(class_name)
            if self.show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            label = ": ".join(label_parts)
            
            if label:
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                label_ymin = max(ymin, label_height + 10)
                cv2.rectangle(frame, (xmin, label_ymin - label_height - 10), (xmin + label_width, label_ymin - baseline), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        cv2.imshow(self.result_win_name, frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CheckerBoard_Detector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()