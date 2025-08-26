#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Header
from custom_interfaces.msg import Ultrasonic

class UltraNode(Node):
    def __init__(self):
        super().__init__('ultra_node')

        # 기존 xycar_ultrasonic 구독
        self.subscription = self.create_subscription(
            Int32MultiArray,
            'xycar_ultrasonic',
            self.ultra_callback,
            10
        )

        # 커스텀 토픽으로 publish
        self.publisher_ = self.create_publisher(Ultrasonic, 'ultrasonic', 10)

        self.ultra_msg = None
        self.get_logger().info("Waiting for ultrasonic data...")
        self.wait_for_message()
        self.get_logger().info("Ultrasonic Ready ----------")

        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz

    def ultra_callback(self, msg):
        self.ultra_msg = msg.data

    def wait_for_message(self):
        while self.ultra_msg is None and rclpy.ok():
            rclpy.spin_once(self)
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.1))

    def timer_callback(self):
        if self.ultra_msg is not None:
            # 커스텀 메시지 생성
            ultra_msg = Ultrasonic()
            ultra_msg.header.stamp = self.get_clock().now().to_msg()
            ultra_msg.header.frame_id = 'ultrasonic'
            ultra_msg.data = self.ultra_msg

            # publish
            self.publisher_.publish(ultra_msg)
            # self.get_logger().info(f"Published Ultrasonic Header: {ultra_msg.header.stamp}")
            # self.get_logger().info(f"Published Ultrasonic Data: {list(ultra_msg.data)}")

def main(args=None):
    rclpy.init(args=args)
    node = UltraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
