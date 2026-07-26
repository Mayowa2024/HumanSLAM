import cv2
import numpy as np
import rclpy
import tensorrt as trt

from collections import deque
from cv_bridge import CvBridge, CvBridgeError
from paddleocr import PaddleOCR
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

from slam.cognitive_math_model import CognitiveMathModel
from slam.types import KeyframeRecord, SceneRecord, StaticObject, TextAnchor


class HumanSLAMNode(Node):
    """
    ROS 2 wrapper for the HumanSLAM semantic recovery model.

    This node:
        1. receives keyframe images
        2. runs scene / YOLO segmentation / OCR processing
        3. builds KeyframeRecord objects
        4. calls CognitiveMathModel
        5. selects a semantic candidate keyframe

    """

    def __init__(self):
        super().__init__("human_slam_node")

        self._declare_parameters()
        self._read_parameters()

        self.bridge = CvBridge()
        self.keyframe_counter = 0

        # Current query scene sequence only.
        self.scene_history = deque(maxlen=3)

        # Full keyframe memory.
        self.map_memory = []

        self.matcher = CognitiveMathModel(
            w_scene=self.w_scene,
            w_object=self.w_object,
            w_text=self.w_text,
            sigma_mask=self.sigma_mask,
            lambda_area=self.lambda_area,
            text_geom_threshold=self.text_geom_threshold,
            semantic_threshold=self.semantic_threshold,
        )

        self.get_logger().info("HumanSLAM node initialising")
        self.get_logger().info(f"Camera topic: {self.camera_source}")

        self._load_models()

        self.image_subscription = self.create_subscription(
            Image,
            self.camera_source,
            self.keyframe_callback,
            10,
        )

        self.get_logger().info("HumanSLAM node initialised")

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        self.declare_parameter("camera_source", "/camera/image_raw")

        self.declare_parameter("scene_classifier_path", "")
        self.declare_parameter("scene_classifier_threshold", 0.5)
        self.declare_parameter("scene_embedding_dim", 512)

        self.declare_parameter("yolo_model_path", "")
        self.declare_parameter("yolo_confidence_threshold", 0.5)

        self.declare_parameter("ocr_confidence_threshold", 0.5)
        self.declare_parameter("ocr_language", "en")
        self.declare_parameter("ocr_use_gpu", True)
        self.declare_parameter("ocr_use_angle_cls", True)
        self.declare_parameter("ocr_show_log", False)

        self.declare_parameter("stable_classes", [])
        self.declare_parameter("crop_margin", 10)
        self.declare_parameter("isolate_mask_for_ocr", True)

        self.declare_parameter("w_scene", 0.3)
        self.declare_parameter("w_object", 0.3)
        self.declare_parameter("w_text", 0.4)
        self.declare_parameter("sigma_mask", 0.25)
        self.declare_parameter("lambda_area", 1.0)
        self.declare_parameter("text_geom_threshold", 0.6)

        self.declare_parameter("semantic_threshold", 0.75)
        self.declare_parameter("geom_inlier_threshold", 30)

    def _read_parameters(self):
        self.camera_source = self.get_parameter("camera_source").value

        self.scene_classifier_path = self.get_parameter("scene_classifier_path").value
        self.scene_classifier_threshold = self.get_parameter(
            "scene_classifier_threshold"
        ).value
        self.scene_embedding_dim = int(self.get_parameter("scene_embedding_dim").value)

        self.yolo_model_path = self.get_parameter("yolo_model_path").value
        self.yolo_conf = float(self.get_parameter("yolo_confidence_threshold").value)

        self.ocr_conf = float(self.get_parameter("ocr_confidence_threshold").value)
        self.ocr_language = self.get_parameter("ocr_language").value
        self.ocr_use_gpu = bool(self.get_parameter("ocr_use_gpu").value)
        self.ocr_use_angle_cls = bool(self.get_parameter("ocr_use_angle_cls").value)
        self.ocr_show_log = bool(self.get_parameter("ocr_show_log").value)

        self.stable_classes = list(self.get_parameter("stable_classes").value)
        self.crop_margin = int(self.get_parameter("crop_margin").value)
        self.isolate_mask_for_ocr = bool(self.get_parameter("isolate_mask_for_ocr").value)

        self.w_scene = float(self.get_parameter("w_scene").value)
        self.w_object = float(self.get_parameter("w_object").value)
        self.w_text = float(self.get_parameter("w_text").value)
        self.sigma_mask = float(self.get_parameter("sigma_mask").value)
        self.lambda_area = float(self.get_parameter("lambda_area").value)
        self.text_geom_threshold = float(self.get_parameter("text_geom_threshold").value)

        self.semantic_threshold = float(self.get_parameter("semantic_threshold").value)
        self.geom_inlier_threshold = int(self.get_parameter("geom_inlier_threshold").value)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        self.scene_engine = None
        self.scene_context = None

        if self.scene_classifier_path:
            try:
                self.scene_engine, self.scene_context = self.load_scene_engine_once(
                    self.scene_classifier_path
                )
                self.get_logger().info("Places365 TensorRT engine loaded")
            except Exception as exc:
                self.get_logger().error(
                    f"Failed to load Places365 TensorRT engine: {exc}"
                )
                # Keep node alive so YOLO/OCR can still be tested.
                self.scene_engine = None
                self.scene_context = None

        if not self.yolo_model_path:
            raise RuntimeError("Parameter yolo_model_path is empty")

        self.yolo = YOLO(self.yolo_model_path)
        self.get_logger().info(f"YOLO model loaded: {self.yolo_model_path}")
        self.get_logger().info(f"YOLO class names: {self.yolo.names}")

        self.ocr_model = PaddleOCR(
            use_angle_cls=self.ocr_use_angle_cls,
            lang=self.ocr_language,
            use_gpu=self.ocr_use_gpu,
            show_log=self.ocr_show_log,
        )
        self.get_logger().info("PaddleOCR model loaded")

    def load_scene_engine_once(self, engine_path):
        """
        Load TensorRT engine once.
        """
        logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as file:
            engine_bytes = file.read()

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)

        if engine is None:
            raise RuntimeError("Failed to deserialize scene classifier engine")

        context = engine.create_execution_context()

        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        return engine, context

    # ------------------------------------------------------------------
    # Keyframe callback
    # ------------------------------------------------------------------

    def keyframe_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"cv_bridge error: {exc}")
            return

        self.keyframe_counter += 1
        keyframe_id = self.keyframe_counter
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        scene_record = self.run_scene_classifier(cv_image)
        self.scene_history.appendleft(scene_record)

        static_objects = self.process_yolo_results(cv_image)

        current_keyframe = KeyframeRecord(
            keyframe_id=keyframe_id,
            timestamp=timestamp,
            scene=scene_record,
            static_objects=static_objects,
            pose=None,
            orb_keyframe_id=None,
            tracking_inliers=None,
        )

        query_scene_seq = list(self.scene_history)
        candidate_keyframes, candidate_scene_sequences = self.build_candidate_inputs()

        if len(candidate_keyframes) > 0:
            best_candidate, semantic_score = self.matcher.select_best_candidate(
                query_keyframe=current_keyframe,
                query_scene_seq=query_scene_seq,
                candidate_keyframes=candidate_keyframes,
                candidate_scene_sequences=candidate_scene_sequences,
            )

            if best_candidate is not None:
                self.get_logger().info(
                    f"Best semantic candidate: {best_candidate.keyframe_id}, "
                    f"score={semantic_score:.3f}"
                )

                # Temporary placeholder until ORB-SLAM3 tracking inliers are connected.
                n_inliers = self.geom_inlier_threshold + 1

                if self.matcher.should_trigger_semantic_recovery(
                    n_inliers=n_inliers,
                    geom_threshold=self.geom_inlier_threshold,
                    semantic_score=semantic_score,
                ):
                    self.semantic_pose_recovery(current_keyframe, best_candidate, semantic_score)

        self.map_memory.append(current_keyframe)

        self.get_logger().info(
            f"Stored keyframe {keyframe_id}: "
            f"{len(static_objects)} static objects, "
            f"{sum(len(obj.texts) for obj in static_objects)} OCR texts"
        )

    # ------------------------------------------------------------------
    # Scene processing
    # ------------------------------------------------------------------

    def run_scene_classifier(self, cv_image) -> SceneRecord:
        """
        TODO:
            Replace placeholder with TensorRT inference.

        For now, this returns a zero vector so the rest of the system can be
        tested with YOLO + OCR + math model.
        """
        if self.scene_engine is None or self.scene_context is None:
            return SceneRecord(
                embedding=np.zeros(self.scene_embedding_dim, dtype=np.float32),
                confidence=0.0,
                label="unknown",
            )

        # TODO: implement preprocess -> TensorRT execute -> output embedding/confidence.
        # Keep placeholder to avoid blocking object/text pipeline development.
        return SceneRecord(
            embedding=np.zeros(self.scene_embedding_dim, dtype=np.float32),
            confidence=0.0,
            label="places365_pending",
        )

    # ------------------------------------------------------------------
    # YOLO segmentation + OCR processing
    # ------------------------------------------------------------------

    def process_yolo_results(self, cv_image):
        h, w, _ = cv_image.shape
        static_objects = []

        yolo_results = self.yolo(
            source=cv_image,
            conf=self.yolo_conf,
            task="segment",
            verbose=False,
        )

        for result in yolo_results:
            boxes = result.boxes
            masks = result.masks

            if boxes is None or masks is None:
                continue

            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.yolo.names[class_id]

                if confidence < self.yolo_conf:
                    continue

                if self.stable_classes and class_name not in self.stable_classes:
                    continue

                mask = masks.data[i].cpu().numpy()
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0.5).astype(np.uint8)

                geometry = self.compute_mask_geometry(mask, w, h)

                if geometry is None:
                    continue

                x_centroid, y_centroid, area = geometry

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                x1 = max(0, x1 - self.crop_margin)
                y1 = max(0, y1 - self.crop_margin)
                x2 = min(w, x2 + self.crop_margin)
                y2 = min(h, y2 + self.crop_margin)

                crop = cv_image[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                if self.isolate_mask_for_ocr:
                    mask_crop = mask[y1:y2, x1:x2]
                    ocr_crop = crop.copy()
                    ocr_crop[mask_crop == 0] = 255
                else:
                    ocr_crop = crop

                texts = self.run_ocr_on_crop(ocr_crop, class_name)

                static_objects.append(
                    StaticObject(
                        class_name=class_name,
                        seg_conf=confidence,
                        x_centroid=x_centroid,
                        y_centroid=y_centroid,
                        area=area,
                        texts=texts,
                    )
                )

                self.get_logger().info(
                    f"Static object: {class_name}, conf={confidence:.2f}, "
                    f"x={x_centroid:.2f}, y={y_centroid:.2f}, area={area:.5f}, "
                    f"texts={len(texts)}"
                )

        return static_objects

    def compute_mask_geometry(self, mask, image_width: int, image_height: int):
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            return None

        x_centroid = float(np.mean(xs) / image_width)
        y_centroid = float(np.mean(ys) / image_height)
        area = float(len(xs) / (image_width * image_height))

        return x_centroid, y_centroid, area

    def run_ocr_on_crop(self, crop, class_name: str):
        detected_texts = []

        try:
            ocr_results = self.ocr_model.ocr(crop, cls=True)
        except Exception as exc:
            self.get_logger().warn(f"OCR failed on {class_name}: {exc}")
            return detected_texts

        if ocr_results is None:
            return detected_texts

        for line_group in ocr_results:
            if line_group is None:
                continue

            for line in line_group:
                try:
                    text = str(line[1][0])
                    confidence = float(line[1][1])
                except Exception:
                    continue

                if confidence >= self.ocr_conf:
                    detected_texts.append(TextAnchor(text=text, conf=confidence))
                    self.get_logger().info(
                        f"OCR on {class_name}: '{text}', conf={confidence:.2f}"
                    )

        return detected_texts

    # ------------------------------------------------------------------
    # Candidate building and recovery
    # ------------------------------------------------------------------

    def build_candidate_inputs(self):
        """
        Returns:
            candidate_keyframes:
                [C_i]

            candidate_scene_sequences:
                [[scene_i, scene_i-1, scene_i-2], ...]
        """
        candidate_keyframes = []
        candidate_scene_sequences = []

        for i, keyframe in enumerate(self.map_memory):
            scene_seq = []

            for j in range(i, max(-1, i - 3), -1):
                scene = self.map_memory[j].scene

                if scene is not None:
                    scene_seq.append(scene)

            candidate_keyframes.append(keyframe)
            candidate_scene_sequences.append(scene_seq)

        return candidate_keyframes, candidate_scene_sequences

    def semantic_pose_recovery(
        self,
        query_keyframe: KeyframeRecord,
        selected_candidate: KeyframeRecord,
        semantic_score: float,
    ):
        """
        Hook for ORB-SLAM3 pose recovery.

        Semantics has selected the keyframe.
        The selected keyframe's geometry/local map should be used for pose adjustment.
        """
        self.get_logger().warn(
            f"Semantic recovery requested: query={query_keyframe.keyframe_id}, "
            f"candidate={selected_candidate.keyframe_id}, score={semantic_score:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = HumanSLAMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
