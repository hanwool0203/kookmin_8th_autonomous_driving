#! /usr/bin/env python
# -*- coding:utf-8 -*-
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import subprocess
import time
import threading
import message_filters

from custom_interfaces.msg import Detections

class CheckerBoard_Detector(Node):

    def __init__(self):
        super().__init__('checkerboard_detector')

        # --- 노드 제어 설정 ---
        self.nodes_to_kill = [
            "scan_rotator_node",
            "pure_pursuit_node",
            "path_planning_node",
            "preprocessing_node"
        ]
        self.is_shutting_down = False # 종료 시퀀스가 중복 실행되는 것을 방지하기 위한 플래그
        # ------------------------------------

        # --- 파라미터 선언 ---
        self.declare_parameter('enable_visualization', False)
        self.declare_parameter('show_class_name', False)
        self.declare_parameter('show_confidence', False)
        self.declare_parameter('bbox_area_threshold', 4000)

        # 파라미터 값 읽어오기
        self.enable_visualization = self.get_parameter('enable_visualization').get_parameter_value().bool_value
        self.show_class_name = self.get_parameter('show_class_name').get_parameter_value().bool_value
        self.show_confidence = self.get_parameter('show_confidence').get_parameter_value().bool_value
        self.bbox_area_threshold = self.get_parameter('bbox_area_threshold').get_parameter_value().integer_value

        # --- 구독자 설정 (동기화) ---
        self.detection_sub = message_filters.Subscriber(self, Detections, '/yolo_detections')
        self.img_sub = message_filters.Subscriber(self, Image, '/image_raw')
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.detection_sub, self.img_sub],
            queue_size=10,
            slop=0.1
        )
        self.time_synchronizer.registerCallback(self.synchronized_callback)

        self.sign_subscription = self.create_subscription(
            String,
            'sign_color',
            self.sign_callback,
            10
        )
        self.is_started = False
        
        # --- 서비스 클라이언트 설정 ---
        self.client = self.create_client(Trigger, 'start_track_driving')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('"/start_track_driving" 서비스를 기다리는 중입니다...')

        self.req = Trigger.Request()

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
    
    def sign_callback(self, msg):
        if self.is_started == True: # 한번 true로 바뀌면 pass
            pass

        if msg.data == 'green':
            self.is_started = True # 초기상태(false)인 경우, 'green'을 받으면 true로 변경

    
    def send_request(self):
        """
        서비스를 비동기적으로 호출하고, 응답이 올 때까지 기다립니다.
        """
        self.get_logger().info('"/start_track_driving" 서비스 호출을 보냅니다...')
        self.future = self.client.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def synchronized_callback(self, detections_msg, image_msg):
        self.detections_callback(detections_msg)
        self.image_callback(image_msg)

    # <<< ======================================================= >>>
    # <<<               핵심 수정: shutdown_sequence 함수             >>>
    # <<< ======================================================= >>>
    def shutdown_sequence(self):
        if self.is_shutting_down:
            return
        self.is_shutting_down = True

        self.get_logger().info("--- 체커보드 감지! 노드 전환 시퀀스를 시작합니다. ---")

        # 1. 종료할 노드들을 개별적으로 종료
        for node_name in self.nodes_to_kill:
            try:
                self.get_logger().info(f"노드 종료 시도: {node_name}")
                result = subprocess.run(f"pkill -f {node_name}", shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    self.get_logger().info(f"'{node_name}' 노드가 성공적으로 종료되었습니다.")
                else:
                    self.get_logger().warn(f"'{node_name}' 노드가 실행 중이 아니거나 이미 종료되었습니다.")
            except Exception as e:
                self.get_logger().error(f"'{node_name}' 노드 종료 중 에러 발생: {e}")

        # 2. [수정됨] 새로운 노드들을 실행하는 대신, 서비스 콜을 통해 활성화 신호를 보냄
        self.get_logger().info("--- 트랙 주행 노드 활성화를 위해 서비스 호출 ---")
        response = self.send_request()
        if response and response.success:
            self.get_logger().info(f"서비스 호출 성공: {response.message}")
        else:
            self.get_logger().error("서비스 호출에 실패했습니다!")

        # 3. 자기 자신 노드 종료
        self.get_logger().info("CheckerBoard_Detector 노드를 종료합니다.")
        if self.enable_visualization:
            cv2.destroyAllWindows()
        self.destroy_node()

    def detections_callback(self, msg):
        if self.is_shutting_down:
            return
        
        if self.is_started == False: #sign_detector가 'green'을 보내기 전까지는 전환 동작 수행 X (노이즈로 인한 전환 방지)
            return

        trigger_shutdown = False
        temp_detections = []

        for detection in msg.detections:
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

            if detection.class_name == 'checkerboard':
                xmin, ymin, xmax, ymax = det_info['bbox']
                area = (xmax - xmin) * (ymax - ymin)
                
                if area >= self.bbox_area_threshold:
                    self.get_logger().info(f"체커보드 감지 (면적: {area} >= 임계값: {self.bbox_area_threshold}). 전환을 시작합니다.")
                    trigger_shutdown = True
                else:
                    self.get_logger().debug(f"체커보드 감지 (면적: {area} < 임계값: {self.bbox_area_threshold}). 무시합니다.")

        with self.lock:
            self.latest_detections = temp_detections

        if trigger_shutdown:
            self.shutdown_sequence()

    def image_callback(self, msg):
        if self.is_shutting_down or not self.enable_visualization:
            return
        
        if self.is_started == False: #sign_detector가 'green'을 보내기 전까지는 전환 동작 수행 X (노이즈로 인한 전환 방지)
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

        # cv2.imshow(self.result_win_name, frame)
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
        # cv2.destroyAllWindows() # destroy_node() 호출 시 자동으로 처리되도록 할 수 있음
        rclpy.shutdown()

if __name__ == '__main__':
    main()