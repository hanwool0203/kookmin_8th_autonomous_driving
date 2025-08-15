import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from ultralytics import YOLO

# 1. 생성한 커스텀 메시지 임포트
from cam_interfaces.msg import Detection, Detections

class Yolo_Node(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # --- 모델 경로 설정 ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(script_dir, 'yolov10n_17000_1.pt')
        self.declare_parameter('model_path', default_model_path)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        
        self.model = YOLO(model_path)
        self.get_logger().info(f"YOLO model loaded successfully from: {model_path}")

        # --- 파라미터 선언 (요청사항) ---
        # 신뢰도 임계값
        self.declare_parameter('class_1_conf_threshold', 0.75)
        self.class_1_conf_threshold = self.get_parameter('class_1_conf_threshold').get_parameter_value().double_value
        self.get_logger().info(f"Confidence threshold for class ID 1 set to: {self.class_1_conf_threshold}")

        # 시각화 옵션
        self.declare_parameter('show_visualization', True)
        self.declare_parameter('show_labels', True)
        self.declare_parameter('show_conf', True)
        #---------------------------------------------


        self.show_visualization = self.get_parameter('show_visualization').get_parameter_value().bool_value
        self.show_labels = self.get_parameter('show_labels').get_parameter_value().bool_value
        self.show_conf = self.get_parameter('show_conf').get_parameter_value().bool_value
        self.get_logger().info(f"Visualization: {self.show_visualization}, Show Labels: {self.show_labels}, Show Conf: {self.show_conf}")


        self.bridge = CvBridge()
        qos_profile = rclpy.qos.qos_profile_sensor_data

        # Subscriber
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos_profile
        )
        
        # Publisher
        self.detection_publisher_ = self.create_publisher(Detections, '/yolo_detections', 10)
        
        self.get_logger().info("YOLO Node has been initialised.")

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # 모델 추론
        results = self.model(cv_image, verbose=False)

        detections_msg = Detections()
        detections_msg.header = msg.header
        
        # 시각화가 활성화된 경우에만 이미지 복사
        annotated_frame = cv_image.copy() if self.show_visualization else None

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # --- class id 1 필터링 로직 ---
            if class_id == 1 and confidence < self.class_1_conf_threshold:
                continue

            # 감지된 객체 정보를 Detection 메시지에 저장 (항상 실행)
            detection = Detection()

            detection.class_name = self.model.names[class_id]
            detection.confidence = confidence
            coords = box.xyxy[0].cpu().numpy().astype(int)
            detection.xmin = int(coords[0])
            detection.ymin = int(coords[1])
            detection.xmax = int(coords[2])
            detection.ymax = int(coords[3])
            
            detections_msg.detections.append(detection)

            # --- 조건부 시각화 로직 (요청사항) ---
            if self.show_visualization:
                # 라벨 문자열 동적 생성
                label_parts = []
                if self.show_labels:
                    label_parts.append(detection.class_name)
                if self.show_conf:
                    label_parts.append(f"{detection.confidence:.2f}")
                label = " ".join(label_parts)
                
                # 클래스 ID에 따라 색상 지정 (BGR)
                if class_id == 0: color = (255, 0, 0)      # Blue
                elif class_id == 1: color = (0, 0, 255)    # Red
                elif class_id == 2: color = (0, 255, 255)  # Yellow
                elif class_id == 3: color = (0, 255, 0)    # Green
                else: color = (128, 128, 128)              # Gray

                # 바운딩 박스 그리기
                cv2.rectangle(annotated_frame, 
                              (detection.xmin, detection.ymin), 
                              (detection.xmax, detection.ymax), 
                              color, 2)
                
                # 라벨이 있을 경우에만 텍스트 그리기
                if label:
                    cv2.putText(annotated_frame, 
                                label, 
                                (detection.xmin, detection.ymin - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 시각화가 활성화된 경우에만 이미지 보여주기
        if self.show_visualization:
            cv2.imshow('yolo_node - annotated_frame', annotated_frame)
            cv2.waitKey(1)

        # 최종 필터링된 메시지 발행
        # 1. 헤더에 현재 시간과 좌표계 정보 채우기
        detections_msg.header.stamp = msg.header.stamp
        detections_msg.header.frame_id = 'camera_link'
        self.detection_publisher_.publish(detections_msg)


def main(args=None):
    rclpy.init(args=args)
    yolo_node = None
    try:
        yolo_node = Yolo_Node()
        rclpy.spin(yolo_node)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 노드 종료 시 창을 닫도록 추가
        cv2.destroyAllWindows()
        if yolo_node:
            yolo_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()