import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from custom_interfaces.msg import ClusterData, ConeData
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import math
import numpy as np 

LEFT_FIRST_CONE_RANGE = 0.8
RIGHT_FIRST_CONE_RANGE = 0.8

LEFT_SECTOR_START, LEFT_SECTOR_END = 30.0, 100.0
RIGHT_SECTOR_START, RIGHT_SECTOR_END = 260.0, 350.0

class RvizVisualizerNode(Node):
    def __init__(self):
        super().__init__('rviz_visualizer_node')
        
        latching_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        default_qos = QoSProfile(depth=10)

        # --- 데이터 구독 ---
        self.cluster_sub = self.create_subscription(ClusterData, '/clusters', self.cluster_callback, 10)
        self.cone_sub = self.create_subscription(ConeData, '/cone_clusters', self.cone_callback, 10)
        self.midpoint_sub = self.create_subscription(ClusterData, '/midpoints', self.midpoint_callback, 10)
        self.target_sub = self.create_subscription(PointStamped, '/target_point', self.target_callback, 10)
        self.motor_sub = self.create_subscription(Float32MultiArray, 'xycar_motor', self.motor_callback,10)
        self.path_sub = self.create_subscription(Path, '/path', self.path_callback, default_qos)

        # --- RViz2 마커 발행 (수정된 QoS 프로파일 적용) ---
        self.cluster_marker_pub = self.create_publisher(MarkerArray, '/visualization/clusters', latching_qos)
        self.left_cone_marker_pub = self.create_publisher(MarkerArray, '/visualization/left_cones', latching_qos)
        self.right_cone_marker_pub = self.create_publisher(MarkerArray, '/visualization/right_cones', latching_qos)
        self.midpoint_marker_pub = self.create_publisher(MarkerArray, '/visualization/midpoints', latching_qos)
        self.target_marker_pub = self.create_publisher(Marker, '/visualization/target_point', latching_qos)
        self.steering_marker_pub = self.create_publisher(Marker, '/visualization/steering', latching_qos)
        self.viz_path_pub = self.create_publisher(Path, '/visualization/path', latching_qos)
        self.sector_marker_pub = self.create_publisher(MarkerArray, '/visualization/sectors', latching_qos)
        self.latest_path_msg = None
        self.visualization_timer = self.create_timer(0.1, self.visualization_timer_callback)
    
    def path_callback(self, msg: Path):
        """고속 경로를 받아서 클래스 변수에 저장하기만 합니다."""
        self.latest_path_msg = msg

    def visualization_timer_callback(self):
        """0.1초마다 호출되어 저장된 최신 경로를 RViz2용 토픽으로 발행합니다."""
        if self.latest_path_msg is not None:
            # 타임스탬프를 현재 시간으로 업데이트하여 발행 (TF 동기화에 도움)
            self.latest_path_msg.header.stamp = self.get_clock().now().to_msg()
            self.viz_path_pub.publish(self.latest_path_msg)

            sector_markers = self.create_sector_markers()
            self.sector_marker_pub.publish(sector_markers)

    def create_sector_markers(self):
        marker_array = MarkerArray()
        # 주의: frame_id는 LiDAR 또는 차량의 기준 좌표계(예: 'base_link')와 일치해야 합니다.
        frame_id = "laser_frame" 
        stamp = self.get_clock().now().to_msg()

        # 부채꼴을 그릴 마커 생성 내부 함수
        def create_marker(sector_id, start_deg, end_deg, color,radius):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = "sectors"
            marker.id = sector_id
            marker.type = Marker.LINE_STRIP  # 라인 스트립 타입
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05  # 라인 두께
            marker.color = color

            points = []
            # 1. 원점 추가
            points.append(Point(x=0.0, y=0.0, z=0.0))
            
            for deg in np.arange(start_deg, end_deg + 1, 5.0):
                rad = np.deg2rad(deg)
                points.append(Point(x=radius * np.cos(rad), y=radius * np.sin(rad), z=0.0))
            
            # 3. 마지막으로 원점을 다시 추가하여 닫힌 도형으로 만듦
            points.append(Point(x=0.0, y=0.0, z=0.0))

            marker.points = points
            return marker

        # 좌측(초록색)과 우측(파란색) 부채꼴 마커 생성
        left_color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
        right_color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
        
        marker_array.markers.append(create_marker(0, LEFT_SECTOR_START, LEFT_SECTOR_END, left_color, LEFT_FIRST_CONE_RANGE))
        marker_array.markers.append(create_marker(1, RIGHT_SECTOR_START, RIGHT_SECTOR_END, right_color, RIGHT_FIRST_CONE_RANGE))

        
        return marker_array

    def create_marker_array(self, points, frame_id, ns, color,scale=0.05, z_offset=0.0):
        marker_array = MarkerArray()
        
        # 이전 마커를 모두 삭제하는 명령을 먼저 보냅니다.
        delete_marker = Marker()
        delete_marker.header.frame_id = frame_id
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # 새로운 마커들을 추가합니다.
        for i, point in enumerate(points):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = ns
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = point
            marker.pose.position.z = z_offset 
            marker.scale.x, marker.scale.y, marker.scale.z = scale, scale, scale 
            marker.color = color
            marker_array.markers.append(marker)
        return marker_array

    def cluster_callback(self, msg: ClusterData):
        markers = self.create_marker_array(msg.clusters, "laser_frame", "clusters", ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0))
        self.cluster_marker_pub.publish(markers)

    def cone_callback(self, msg: ConeData):
        left_markers = self.create_marker_array(msg.left_cones, "laser_frame", "left_cones", ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),scale=0.08, z_offset=0.12)
        right_markers = self.create_marker_array(msg.right_cones, "laser_frame", "right_cones", ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),scale=0.08, z_offset=0.12)
        self.left_cone_marker_pub.publish(left_markers)
        self.right_cone_marker_pub.publish(right_markers)

    def midpoint_callback(self, msg: ClusterData):
        markers = self.create_marker_array(msg.clusters, "laser_frame", "midpoints", ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0))
        self.midpoint_marker_pub.publish(markers)

    def target_callback(self, msg: PointStamped):
        marker = Marker()
        marker.header.frame_id = msg.header.frame_id
        marker.header.stamp = msg.header.stamp
        marker.ns = "target_point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = msg.point
        marker.pose.position = msg.point
        marker.pose.position.z = 0.2
        marker.scale.x, marker.scale.y, marker.scale.z = 0.05,0.05,0.05
        marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)
        self.target_marker_pub.publish(marker)

    def motor_callback(self, msg: Float32MultiArray):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "steering"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        start_point, end_point = Point(), Point()
        original_angle_cmd = -msg.data[0]
        restored_rad = math.radians(original_angle_cmd * 0.2)
        end_point.x = 1.0 * math.cos(restored_rad) 
        end_point.y = 1.0 * math.sin(restored_rad)
        marker.points = [start_point, end_point]
        marker.scale.x, marker.scale.y, marker.scale.z = 0.05, 0.1, 0.1
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        self.steering_marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = RvizVisualizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
