import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import os
import json
import cv2
import requests
from ultralytics import YOLO

from px4_msgs.msg import VehicleLocalPosition

try:
    from resource_probe import stamp as stamp_resources  # GPU/CPU/RAM/net on drone.frame
except Exception:  # keep the node working if the probe (or psutil/pynvml) is absent
    def stamp_resources(span):
        return None

# OpenTelemetry — set up at module load so RequestsInstrumentor wraps every
# outbound POST. The W3C traceparent header is injected automatically, so the
# backend's /receive_detection span chains to ours.
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor

SERVICE_NAME = "drone_yolo_service"
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT_GRPC")

# Swarm identity. DRONE_ID labels every detection/span; DRONE_NS is the PX4
# uXRCE-DDS namespace (e.g. "px4_1") so a per-drone node subscribes to that
# vehicle's topics. Defaults reproduce the single-drone (un-namespaced) path.
DRONE_ID = int(os.getenv("DRONE_ID", "1"))
DRONE_NS = os.getenv("DRONE_NS", "").strip("/")

_resource = Resource(attributes={"service.name": SERVICE_NAME, "drone.id": DRONE_ID})
_provider = TracerProvider(resource=_resource)
if OTEL_ENDPOINT:
    _provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True))
    )
trace.set_tracer_provider(_provider)
tracer = trace.get_tracer(__name__)
RequestsInstrumentor().instrument()

class UAVCameraDetector(Node):
    def __init__(self):
        super().__init__(f'uav_camera_detector_{DRONE_ID}')
        self.bridge = CvBridge()
        self.drone_id = DRONE_ID

        # Per-drone topic prefix. Empty namespace keeps the legacy relative
        # 'camera' / '/fmu/...' topics; a namespace (e.g. px4_1) scopes both.
        pfx = f"/{DRONE_NS}" if DRONE_NS else ""
        camera_topic = os.getenv("CAMERA_TOPIC") or (f"{pfx}/camera" if DRONE_NS else "camera")
        lpos_topic = f"{pfx}/fmu/out/vehicle_local_position"

        # Load YOLO model
        self.model = YOLO('yolov8n.pt')  # Adjust path if needed

        # Subscribe to ROS image topic
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10)
        self.get_logger().info(
            f"YOLO node (drone {DRONE_ID}, ns='{DRONE_NS or '-'}') subscribed to '{camera_topic}'")

        # Backend detection receiver endpoint
        self.backend_origin = os.getenv('BACKEND_ORIGIN')
        self.backend_url = self.backend_origin + '/receive_detection'
        self.get_logger().info(f"Backend URL set to: {self.backend_url}")

        self.show_window = os.getenv('UAV_CAMERA_SHOW', '1') != '0'
        if self.show_window:
            cv2.namedWindow('UAV YOLO', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('UAV YOLO', 960, 540)

        # Latest drone NED pose (None until first lpos arrives).
        self.pose = None
        lpos_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            VehicleLocalPosition,
            lpos_topic,
            self._on_lpos,
            lpos_qos,
        )

    def _on_lpos(self, msg):
        if not (msg.xy_valid and msg.z_valid):
            return
        self.pose = {
            "n": float(msg.x),
            "e": float(msg.y),
            "d": float(msg.z),
            "yaw": float(msg.heading),
            "valid": True,
        }

    def image_callback(self, msg):
        with tracer.start_as_current_span("drone.frame") as frame_span:
            frame_span.set_attribute("drone.id", self.drone_id)
            stamp_resources(frame_span)  # shared-GPU util/mem + host CPU/RAM/net
            self.get_logger().info("Image received")

            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                frame_span.set_attribute("outcome", "cv_bridge_error")
                frame_span.record_exception(e)
                self.get_logger().error(f"cv_bridge conversion failed: {e}")
                return

            try:
                with tracer.start_as_current_span("drone.yolo_inference") as inf_span:
                    start_time = time.time()
                    results = self.model(cv_image)
                    end_time = time.time()
                    inf_span.set_attribute("inference.duration_ms", round((end_time - start_time) * 1000, 2))
                self.get_logger().info("YOLO inference completed")
            except Exception as e:
                frame_span.set_attribute("outcome", "yolo_error")
                frame_span.record_exception(e)
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

            frame_span.set_attribute("detection.count", len(formatted_detections))

            payload = {
                "drone_id": self.drone_id,
                "timestamp": time.time(),
                "inference_time_ms": round((end_time - start_time) * 1000, 2),
                "speed": {
                    "preprocess": round(speed_info['preprocess'], 2),
                    "inference": round(speed_info['inference'], 2),
                    "postprocess": round(speed_info['postprocess'], 2)
                },
                "pose": self.pose,
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
                frame_span.set_attribute("backend.status_code", response.status_code)
                frame_span.set_attribute("outcome", "posted")
                self.get_logger().info(f"Backend response: {response.status_code} - {response.text}")
            except Exception as e:
                frame_span.set_attribute("outcome", "post_error")
                frame_span.record_exception(e)
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
