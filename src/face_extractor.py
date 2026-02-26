"""
Módulo para detectar y extraer rostros de videos.
Ofrece tres backends: OpenCV (Haar Cascade), MTCNN y MediaPipe.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import os


class FaceDetector(ABC):
    """Clase base abstracta para detectores de rostros."""

    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detecta rostros en una imagen.

        Args:
            image: Imagen en formato BGR (OpenCV)

        Returns:
            Lista de diccionarios con 'box' (x, y, w, h) y 'confidence'
        """
        pass


class OpenCVDetector(FaceDetector):
    """
    Detector de rostros usando OpenCV Haar Cascade.
    Usa múltiples cascades para mejor detección.
    """

    def __init__(
        self,
        scale_factor: float = 1.05,
        min_neighbors: int = 3,
        min_face_size: int = 20,
    ):
        # Cargar múltiples clasificadores para mejor cobertura
        self.cascades = []

        cascade_files = [
            "haarcascade_frontalface_alt2.xml",  # Mejor para rostros variados
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt.xml",
        ]

        for cascade_file in cascade_files:
            cascade_path = cv2.data.haarcascades + cascade_file
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                self.cascades.append(cascade)

        if not self.cascades:
            raise RuntimeError("No se pudo cargar ningún clasificador Haar Cascade")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Mejorar contraste para mejor detección
        gray = cv2.equalizeHist(gray)

        all_faces = []

        # Probar con cada cascade
        for cascade in self.cascades:
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(self.min_face_size, self.min_face_size),
            )

            for x, y, w, h in faces:
                all_faces.append(
                    {"box": [int(x), int(y), int(w), int(h)], "confidence": 0.85}
                )

            # Si encontramos rostros, no seguir buscando
            if len(faces) > 0:
                break

        # Eliminar detecciones duplicadas (boxes muy similares)
        if len(all_faces) > 1:
            all_faces = self._remove_duplicates(all_faces)

        return all_faces

    def _remove_duplicates(
        self, faces: List[dict], iou_threshold: float = 0.5
    ) -> List[dict]:
        """Elimina detecciones duplicadas basándose en IoU."""
        if not faces:
            return faces

        # Ordenar por área (más grandes primero)
        faces = sorted(faces, key=lambda f: f["box"][2] * f["box"][3], reverse=True)

        keep = []
        for face in faces:
            is_duplicate = False
            for kept in keep:
                if self._calculate_iou(face["box"], kept["box"]) > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(face)

        return keep

    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calcula Intersection over Union entre dos boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calcular intersección
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter_area = (xi2 - xi1) * (yi2 - yi1)

        # Calcular unión
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


class MTCNNDetector(FaceDetector):
    """Detector de rostros usando MTCNN."""

    def __init__(self, min_face_size: int = 40):
        from mtcnn import MTCNN

        self.detector = MTCNN(min_face_size=min_face_size)

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        # MTCNN espera RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb_image)

        results = []
        for det in detections:
            results.append(
                {
                    "box": det["box"],  # [x, y, width, height]
                    "confidence": det["confidence"],
                }
            )
        return results


class MediaPipeDetector(FaceDetector):
    """
    Detector de rostros usando MediaPipe.
    Soporta tanto la API legacy (solutions) como la nueva API (tasks).
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        self.min_confidence = min_detection_confidence
        self.detector = None
        self.use_legacy_api = False
        self.use_tasks_api = False

        # Intentar cargar MediaPipe (primero legacy, luego tasks)
        try:
            import mediapipe as mp

            # Intentar API legacy (mp.solutions) - versiones < 0.10.31
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                self.mp = mp
                self.mp_face_detection = mp.solutions.face_detection
                self.detector = self.mp_face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=min_detection_confidence
                )
                self.use_legacy_api = True
                print("MediaPipe: usando API legacy (solutions)")

            # Intentar API nueva (tasks) - versiones >= 0.10.31
            elif hasattr(mp, "tasks"):
                self._setup_tasks_api(mp, min_detection_confidence)
                self.use_tasks_api = True
                print("MediaPipe: usando API nueva (tasks)")

            else:
                raise ImportError("MediaPipe no tiene 'solutions' ni 'tasks'")

        except Exception as e:
            raise ImportError(
                f"Error cargando MediaPipe: {e}\n"
                "Opciones:\n"
                "  1. Instalar versión compatible: pip install mediapipe==0.10.9\n"
                "  2. Usar detector 'opencv' o 'mtcnn' en su lugar"
            )

    def _setup_tasks_api(self, mp, min_confidence):
        """Configura la API nueva de MediaPipe Tasks."""
        import urllib.request

        # Descargar modelo si no existe
        model_path = Path(__file__).parent / "blaze_face_short_range.tflite"
        if not model_path.exists():
            print("Descargando modelo de MediaPipe...")
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(url, str(model_path))

        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        self.detector = FaceDetector.create_from_options(options)
        self.mp = mp

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        if self.use_legacy_api:
            return self._detect_legacy(image)
        elif self.use_tasks_api:
            return self._detect_tasks(image)
        return []

    def _detect_legacy(self, image: np.ndarray) -> List[dict]:
        """Detección usando API legacy (solutions)."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_image)

        detections = []
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                detections.append(
                    {"box": [x, y, width, height], "confidence": detection.score[0]}
                )

        return detections

    def _detect_tasks(self, image: np.ndarray) -> List[dict]:
        """Detección usando API nueva (tasks)."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_image)

        results = self.detector.detect(mp_image)

        detections = []
        h, w = image.shape[:2]
        for detection in results.detections:
            bbox = detection.bounding_box
            detections.append(
                {
                    "box": [bbox.origin_x, bbox.origin_y, bbox.width, bbox.height],
                    "confidence": detection.categories[0].score,
                }
            )

        return detections


class FaceExtractor:
    """Clase principal para extraer rostros de videos."""

    def __init__(
        self,
        detector_type: str = "opencv",
        min_face_size: int = 80,
        output_size: Tuple[int, int] = (224, 224),
        min_confidence: float = 0.7,
        padding: float = 0.2,
    ):
        """
        Args:
            detector_type: "opencv" (sin dependencias extra), "mediapipe" o "mtcnn"
            min_face_size: Tamaño mínimo de rostro a detectar (pixeles)
            output_size: Tamaño de salida de las imágenes de rostros
            min_confidence: Confianza mínima para aceptar una detección
            padding: Porcentaje de padding alrededor del rostro (0.2 = 20%)
        """
        self.min_face_size = min_face_size
        self.output_size = output_size
        self.min_confidence = min_confidence
        self.padding = padding

        if detector_type == "opencv":
            self.detector = OpenCVDetector(min_face_size=min_face_size)
        elif detector_type == "mediapipe":
            self.detector = MediaPipeDetector(min_detection_confidence=min_confidence)
        elif detector_type == "mtcnn":
            self.detector = MTCNNDetector(min_face_size=min_face_size)
        else:
            raise ValueError(
                f"Detector no soportado: {detector_type}. Usa 'opencv', 'mediapipe' o 'mtcnn'"
            )

    def extract_face(self, image: np.ndarray, box: List[int]) -> Optional[np.ndarray]:
        """
        Extrae y redimensiona un rostro de la imagen.

        Args:
            image: Imagen original
            box: [x, y, width, height] del rostro

        Returns:
            Imagen del rostro recortada y redimensionada
        """
        x, y, w, h = box
        img_h, img_w = image.shape[:2]

        # Filtrar rostros muy pequeños
        if w < self.min_face_size or h < self.min_face_size:
            return None

        # Agregar padding
        pad_w = int(w * self.padding)
        pad_h = int(h * self.padding)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)

        # Recortar rostro
        face = image[y1:y2, x1:x2]

        if face.size == 0:
            return None

        # Redimensionar
        face = cv2.resize(face, self.output_size, interpolation=cv2.INTER_LANCZOS4)

        return face

    def process_frame(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Procesa un frame y extrae todos los rostros detectados.

        Args:
            frame: Frame del video (BGR)

        Returns:
            Lista de imágenes de rostros
        """
        detections = self.detector.detect_faces(frame)
        faces = []

        for det in detections:
            if det["confidence"] >= self.min_confidence:
                face = self.extract_face(frame, det["box"])
                if face is not None:
                    faces.append(face)

        return faces

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        sample_interval: float = 0.5,
        max_faces_per_video: int = 100,
    ) -> int:
        """
        Procesa un video y guarda los rostros extraídos.

        Args:
            video_path: Ruta al archivo de video
            output_dir: Directorio donde guardar los rostros
            sample_interval: Intervalo en segundos entre muestreos
            max_faces_per_video: Máximo de rostros a extraer por video

        Returns:
            Número de rostros extraídos
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_name = Path(video_path).stem

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: No se puede abrir el video {video_path}")
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = max(1, int(fps * sample_interval))

        print(f"Procesando: {video_name}")
        print(
            f"  FPS: {fps:.2f}, Frames totales: {total_frames}, Intervalo: cada {sample_rate} frames"
        )

        face_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Muestrear frames según intervalo
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue

            # Detectar y extraer rostros
            faces = self.process_frame(frame)

            for face in faces:
                if face_count >= max_faces_per_video:
                    break

                # Guardar rostro
                face_filename = (
                    f"{video_name}_frame{frame_idx:06d}_face{face_count:04d}.jpg"
                )
                face_path = output_dir / face_filename
                cv2.imwrite(str(face_path), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
                face_count += 1

            if face_count >= max_faces_per_video:
                print(f"  Alcanzado límite de {max_faces_per_video} rostros")
                break

            frame_idx += 1

        cap.release()
        print(f"  Rostros extraídos: {face_count}")
        return face_count


def check_face_quality(face_image: np.ndarray) -> dict:
    """
    Evalúa la calidad de una imagen de rostro.

    Args:
        face_image: Imagen del rostro

    Returns:
        Diccionario con métricas de calidad
    """
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # Calcular nitidez usando varianza del Laplaciano
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Calcular brillo promedio
    brightness = np.mean(gray)

    # Calcular contraste
    contrast = np.std(gray)

    return {
        "sharpness": laplacian_var,
        "brightness": brightness,
        "contrast": contrast,
        "is_good_quality": laplacian_var > 100
        and 50 < brightness < 200
        and contrast > 30,
    }


if __name__ == "__main__":
    # Ejemplo de uso
    extractor = FaceExtractor(
        detector_type="mediapipe", output_size=(224, 224), min_confidence=0.7
    )

    # Probar con una imagen
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = extractor.process_frame(test_image)
    print(f"Rostros detectados en imagen de prueba: {len(faces)}")
