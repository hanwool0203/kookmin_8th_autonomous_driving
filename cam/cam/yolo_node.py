import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from ultralytics import YOLO
from collections import defaultdict

# 1. 생성한 커스텀 메시지 임포트
from custom_interfaces.msg import Detection, Detections

class Yolo_Node(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # --- 토픽 이름 설정 ---
        self.declare_parameter('image_topic_name', '/image_raw') # 이전 백파일 불러올 때는 '/stamped_image_raw'로 변경하여 사용
        ImageTopicName = self.get_parameter('image_topic_name').get_parameter_value().string_value
        self.get_logger().info(f"Image topic name set to: {ImageTopicName}")

        # --- 모델 경로 설정 ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(script_dir, 'yolov10n_20000.pt')
        self.declare_parameter('model_path', default_model_path)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        
        self.model = YOLO(model_path)
        self.get_logger().info(f"YOLO model loaded successfully from: {model_path}")

        # --- 파라미터 선언 (요청사항) ---
        # 신뢰도 임계값
        self.declare_parameter('class_1_conf_threshold', 0.1) #0.75
        self.class_1_conf_threshold = self.get_parameter('class_1_conf_threshold').get_parameter_value().double_value
        self.get_logger().info(f"Confidence threshold for class ID 1 set to: {self.class_1_conf_threshold}")

        # 시각화 옵션
        self.declare_parameter('show_visualization', True)
        self.declare_parameter('show_labels', False)
        self.declare_parameter('show_conf', True)
        
        # --- [추가] 겹치는 박스 제거를 위한 IoU 임계값 ---
        self.declare_parameter('iou_threshold', 0.6)
        self.iou_threshold = self.get_parameter('iou_threshold').get_parameter_value().double_value
        self.get_logger().info(f"IoU threshold for NMS set to: {self.iou_threshold}")
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
            ImageTopicName,
            self.image_callback,
            qos_profile
        )
        
        # Publisher
        self.detection_publisher_ = self.create_publisher(Detections, '/yolo_detections', 10)
        
        self.get_logger().info("YOLO Node has been initialised.")

    def calculate_iou(self, box1, box2):
        """두 바운딩 박스의 IoU(Intersection over Union)를 계산합니다."""
        # box1, box2는 Detection 메시지 객체
        xmin1, ymin1, xmax1, ymax1 = box1.xmin, box1.ymin, box1.xmax, box1.ymax
        xmin2, ymin2, xmax2, ymax2 = box2.xmin, box2.ymin, box2.xmax, box2.ymax

        # 교차 영역 계산
        inter_xmin = max(xmin1, xmin2)
        inter_ymin = max(ymin1, ymin2)
        inter_xmax = min(xmax1, xmax2)
        inter_ymax = min(ymax1, ymax2)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

        # 각 박스 영역 계산
        box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
        box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

        # 합집합 영역 계산
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        iou = inter_area / union_area
        return iou

    def apply_class_based_nms(self, detections):
        """동일 클래스 내에서 겹치는 박스를 제거합니다 (신뢰도 높은 박스 유지)."""
        
        # 클래스 이름별로 detection 그룹화
        grouped_detections = defaultdict(list)
        for det in detections:
            grouped_detections[det.class_name].append(det)

        final_detections = []
        for class_name, dets in grouped_detections.items():
            # 신뢰도 기준으로 내림차순 정렬
            dets.sort(key=lambda x: x.confidence, reverse=True)
            
            kept_dets = []
            while dets:
                # 가장 신뢰도 높은 박스를 선택하고 kept 리스트에 추가
                best_det = dets.pop(0)
                kept_dets.append(best_det)
                
                # 남은 박스들과 IoU 비교
                remaining_dets = []
                for det in dets:
                    iou = self.calculate_iou(best_det, det)
                    # IoU가 임계값 미만인 박스만 남김
                    if iou < self.iou_threshold:
                        remaining_dets.append(det)
                
                # 다음 순회를 위해 리스트 업데이트
                dets = remaining_dets
            
            final_detections.extend(kept_dets)
            
        return final_detections

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # 모델 추론
        results = self.model(cv_image, verbose=False)

        # --- [변경] 1. 모든 후보 detection을 리스트에 먼저 저장 ---
        candidate_detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if class_id == 1 and confidence < self.class_1_conf_threshold:
                continue

            detection = Detection()
            detection.class_name = self.model.names[class_id]
            detection.confidence = confidence
            coords = box.xyxy[0].cpu().numpy().astype(int)
            detection.xmin = int(coords[0])
            detection.ymin = int(coords[1])
            detection.xmax = int(coords[2])
            detection.ymax = int(coords[3])
            
            candidate_detections.append(detection)

        # --- [추가] 2. 후보 detection 리스트에 NMS 후처리 적용 ---
        final_detections = self.apply_class_based_nms(candidate_detections)

        # --- [변경] 3. 최종 필터링된 detection으로 메시지 생성 및 시각화 ---
        detections_msg = Detections()
        detections_msg.header = msg.header
        annotated_frame = cv_image.copy() if self.show_visualization else None

        for detection in final_detections:
            detections_msg.detections.append(detection)

            if self.show_visualization:
                label_parts = []
                if self.show_labels:
                    label_parts.append(detection.class_name)
                if self.show_conf:
                    label_parts.append(f"{detection.confidence:.2f}")
                label = " ".join(label_parts)
                
                # 클래스 이름을 기반으로 색상 가져오기 (ID 대신)
                class_id = list(self.model.names.keys())[list(self.model.names.values()).index(detection.class_name)]
                if class_id == 0: color = (255, 0, 0)      # Blue
                elif class_id == 1: color = (0, 0, 255)    # Red
                elif class_id == 2: color = (0, 255, 255)  # Yellow
                elif class_id == 3: color = (0, 255, 0)    # Green
                else: color = (128, 128, 128)              # Gray

                cv2.rectangle(annotated_frame, 
                              (detection.xmin, detection.ymin), 
                              (detection.xmax, detection.ymax), 
                              color, 2)
                
                if label:
                    cv2.putText(annotated_frame, 
                                label, 
                                (detection.xmin, detection.ymin - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self.show_visualization:
            # cv2.imshow('yolo_node - annotated_frame', annotated_frame)
            cv2.waitKey(1)

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
        cv2.destroyAllWindows()
        if yolo_node:
            yolo_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()