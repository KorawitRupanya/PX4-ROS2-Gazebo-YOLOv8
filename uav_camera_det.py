import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import os
import json
import cv2
import requests
from ultralytics import YOLO

class UAVCameraDetector(Node):
    def __init__(self):
        super().__init__('uav_camera_detector')
        self.bridge = CvBridge()

        # Load YOLO model
        self.model = YOLO('yolov8n.pt')  # Adjust path if needed

        # Subscribe to ROS image topic
        self.subscription = self.create_subscription(
            Image,
            'camera',  # Adjust topic name if needed
            self.image_callback,
            10)
        self.get_logger().info("YOLO node initialized and subscribed to 'camera' topic")

        # Backend detection receiver endpoint
        self.backend_origin = os.getenv('BACKEND_ORIGIN')
        self.backend_url = self.backend_origin + '/receive_detection'
        self.get_logger().info(f"Backend URL set to: {self.backend_url}")

        self.show_window = os.getenv('UAV_CAMERA_SHOW', '1') != '0'
        if self.show_window:
            cv2.namedWindow('UAV YOLO', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('UAV YOLO', 960, 540)

    def image_callback(self, msg):
        self.get_logger().info("Image received")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        try:
            start_time = time.time()
            results = self.model(cv_image)
            end_time = time.time()
            self.get_logger().info("YOLO inference completed")
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        result = results[0]
        speed_info = result.speed
        detections = result.boxes

        formatted_detections = []
        if detections and len(detections.xyxy) > 0:
            for i, box in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(float, box[:4])
                conf = float(detections.conf[i])
                cls = int(detections.cls[i])
                label = self.model.names.get(cls, f"class_{cls}")
                formatted_detections.append({
                    "class_id": cls,
                    "class_name": label,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2]
                })

        payload = {
            "drone_id": 1,
            "timestamp": time.time(),
            "inference_time_ms": round((end_time - start_time) * 1000, 2),
            "speed": {
                "preprocess": round(speed_info['preprocess'], 2),
                "inference": round(speed_info['inference'], 2),
                "postprocess": round(speed_info['postprocess'], 2)
            },
            "detections": formatted_detections
        }

        if self.show_window:
            annotated = result.plot()
            cv2.imshow('UAV YOLO', annotated)
            cv2.waitKey(1)

        self.get_logger().info("Sending detection payload to backend...")
        self.get_logger().debug(json.dumps(payload, indent=2))

        try:
            response = requests.post(self.backend_url, json=payload, timeout=2)
            self.get_logger().info(f"Backend response: {response.status_code} - {response.text}")
        except Exception as e:
            self.get_logger().error(f"Failed to send detection to backend: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = UAVCameraDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received, shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
