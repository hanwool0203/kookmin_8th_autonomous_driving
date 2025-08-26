#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from custom_interfaces.msg import Detections
import message_filters

class SignDetector(Node):
    """
    카메라 이미지와 YOLO 탐지 결과를 받아 신호등을 감지하고, 녹색불일 경우
    미션을 완료한 뒤 안전하게 종료되는 ROS2 노드.
    """
    def __init__(self):
        super().__init__('sign_detector')

        self.context.on_shutdown(self.cleanup)

        # 카메라 이미지 토픽 구독 설정
        # --- 구독 동기화 ---
        self.image_subscription = message_filters.Subscriber(self,Image,'/image_raw') # 이전 백파일 사용할 때는 '/stamped_image_raw'로 변경
        self.yolo_subscription = message_filters.Subscriber(self,Detections,'/yolo_detections')

        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.image_subscription, self.yolo_subscription],
            queue_size=10,
            slop=0.1
        )
        self.time_synchronizer.registerCallback(self.synchronized_callback)
        # -----------------

        self.sign_color_publisher = self.create_publisher(String, '/sign_color', 10)

        # 클래스 변수 초기화
        self.bridge = CvBridge()
        self.mission_triggered = False
        self.showImage = True
        self.frame = None
        self.traffic_light_box = None # 신호등 바운딩 박스 좌표를 저장할 변수

        if self.showImage:
            self.window_name = 'Sign Detector'
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self.hsv_callback_method)

        self.get_logger().info('SignDetector has been started.')

    # --- 콜백 동기화 ---
    def synchronized_callback(self, image_msg, yolo_msg):
            #self.get_logger().info('<<<<< SYNCHRONIZED CALLBACK IS RUNNING! >>>>>') # 콜백 호출 디버깅
            self.yolo_callback(yolo_msg)
            self.image_callback(image_msg)
    # -----------------

    def cleanup(self):
        """노드 종료 시 모든 리소스를 정리하는 단일 메서드."""
        print("\n--- Starting Cleanup Process ---")
        if self.showImage:
            print("Closing OpenCV windows...")
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        print("--- Cleanup Process Finished ---")

    # [수정 3] yolo_detections 토픽을 처리할 새로운 콜백 함수
    def yolo_callback(self, msg):
        """
        /yolo_detections 토픽을 받아 'race_traffic_light' 클래스를 찾고
        해당 바운딩 박스 정보를 저장합니다.
        """
        # 매번 새로운 메시지를 받을 때마다 초기화
        self.traffic_light_box = None
        for detection in msg.detections:
            if detection.class_name == 'race_traffic_light':
                # 필요한 좌표들을 튜플 형태로 저장
                self.traffic_light_box = (
                    detection.xmin,
                    detection.ymin,
                    detection.xmax,
                    detection.ymax
                )
                # 첫 번째 신호등을 찾으면 반복 중단 (필요에 따라 로직 수정 가능)
                break

    def image_callback(self, msg):
        """카메라 이미지를 받아 신호등을 감지하는 메인 콜백 함수."""
        if self.mission_triggered or not rclpy.ok():
            return
            
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        roi = None
        # 시각화를 위해 원본 이미지를 복사
        result_img = frame.copy()

        # [수정 4] yolo_callback에서 저장한 바운딩 박스 정보 사용
        if self.traffic_light_box is not None:
            # 저장된 좌표를 정수형으로 변환하여 사용
            x1, y1, x2, y2 = map(int, self.traffic_light_box)
            
            # 좌표를 이용해 ROI(Region of Interest) 추출
            # 좌표가 이미지 범위를 벗어나지 않도록 클램핑
            y1, y2 = max(0, y1), min(frame.shape[0], y2)
            x1, x2 = max(0, x1), min(frame.shape[1], x2)
            roi = frame[y1:y2, x1:x2]

            # 시각화를 위해 결과 이미지에 바운딩 박스 그리기
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_img, 'race_traffic_light', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if roi is not None and roi.size > 0:
            color = self.detect_green_light(roi)
            if color == 'green' and not self.mission_triggered:
                self.mission_triggered = True
                self.get_logger().info('<<<<< GREEN LIGHT DETECTED! Shutting down node. >>>>>')
                
                color_msg = String()
                
                color_msg.data = color
                self.sign_color_publisher.publish(color_msg)
                
                rclpy.shutdown()

        if rclpy.ok():
            # 원본 프레임 대신 바운딩 박스가 그려진 result_img를 시각화에 사용
            self.visualizing(result_img, roi, show=self.showImage)

    def hsv_callback_method(self, event, x, y, flags, param):
        """마우스 클릭으로 BGR 및 HSV 값을 확인하는 디버깅용 함수."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.frame is None:
                print("Image not available yet.")
                return
            bgr_pixel = self.frame[y, x]
            hsv_pixel = cv2.cvtColor(np.uint8([[bgr_pixel]]), cv2.COLOR_BGR2HSV)[0][0]
            print(f"Clicked (x,y): ({x},{y}) | BGR: {bgr_pixel} | HSV: {hsv_pixel}")

    def detect_green_light(self, image):
        """ROI 이미지에서 녹색 신호등을 검출하는 함수."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 녹색 검출을 위한 HSV 범위 (환경에 따라 조정 필요)
        green_lower = np.array([70, 100, 120])
        green_upper = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # ROI 크기에 비례하여 임계값 설정 (예: ROI 면적의 5%)
        threshold = (image.shape[0] * image.shape[1]) * 0.05 
        
        if cv2.countNonZero(mask) > 100:
            return "green"
        return None

    def visualizing(self, frame_to_show, roi, show=False):
        """결과 이미지를 화면에 표시하는 함수."""
        if show:
            self.frame = frame_to_show # 마우스 콜백에서 사용하기 위해 현재 프레임 저장
            cv2.imshow(self.window_name, frame_to_show)
            if roi is not None:
                cv2.imshow('ROI', roi)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = SignDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt detected, shutting down.')
    finally:
        if node.context.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()