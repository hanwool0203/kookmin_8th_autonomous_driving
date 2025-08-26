import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class TopicStamper(Node):
    def __init__(self):
        super().__init__('topic_stamper')

        # 왜곡 보정 실행 여부
        self.UNDISTORTING = True

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
        self.stamped_image_publisher = self.create_publisher(Image, '/stamped_image_raw', 10)

        # ✅ 왜곡 보정용 내부 파라미터 설정
        self.img_size = (640, 480)
        self.mtx = np.array([[329.7890410051319, 0.000000, 320.6057545861696],[0.000000, 329.76662478183584, 229.13605526487663],[0.000000, 0.000000, 1.000000]])
        self.dist = np.array([-0.00845590148493767, -0.020724735072096084, 0.007699567026492894, 0.004404855849468008])
        self.cal_mtx = self.mtx
        self.new_mtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.mtx, self.dist, self.img_size, np.eye(3), balance=0)
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.mtx, self.dist, np.eye(3), self.new_mtx, self.img_size, cv2.CV_32FC1)
        
        self.get_logger().info("Topic Stamper has been initialised.")

    ### ===== 어안 렌즈 이미지의 왜곡을 보정한다 =====
    def to_calibrated(self, img):
        # 어안 렌즈 이미지 왜곡 보정
        # cv2.remap 함수와 initUndistortRectifyMap에서 계산된 맵을 사용하여 왜곡 보정 수행
        img_undistorted = cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return img_undistorted

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
        
        if self.UNDISTORTING:
            # 왜곡 보정
            cv_image = self.to_calibrated(cv_image)

        # 최종 필터링된 메시지 발행
        # 1. 헤더에 현재 시간과 좌표계 정보 채우기
        stamped_image_msg = Image()
        stamped_image_msg.header.stamp = self.get_clock().now().to_msg()
        stamped_image_msg.header.frame_id = 'camera_link'

        stamped_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")

        self.stamped_image_publisher.publish(stamped_image_msg)


def main(args=None):
    rclpy.init(args=args)
    yolo_node = None
    try:
        yolo_node = TopicStamper()
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