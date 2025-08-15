import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

class ClickSaver(Node):
    def __init__(self):
        super().__init__('click_saver')
        self.subscription = self.create_subscription(
            PointStamped,
            '/clicked_point',  # RViz2에서 자동 출력되는 토픽명
            self.callback,
            10
        )
        self.clicked_points = []

    def callback(self, msg):
        pt = [msg.point.x, msg.point.y, msg.point.z]
        print(f"Clicked: {pt}")
        self.clicked_points.append(pt)
        # 자동 저장 예시 (원하면 파일 저장 추가)
        with open('clicked_points.csv', 'a') as f:
            f.write(f"{pt[0]},{pt[1]},{pt[2]}\n")

def main():
    rclpy.init()
    node = ClickSaver()
    print("클릭할 때마다 좌표가 clicked_points.csv에 저장됩니다.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
