"""
Modulo para detectar, alinear y extraer rostros de videos e imagenes.

Backend principal: MediaPipe Tasks (FaceDetector + FaceLandmarker)
- Deteccion precisa con modelo BlazeFace
- Alineacion facial usando landmarks (ojos) para normalizar rotacion
- Padding proporcional y verificacion de calidad estricta
- Sin rostros incompletos: se descarta cualquier recorte que toque el borde

Fallback: OpenCV Haar Cascade (si MediaPipe no esta disponible)
"""

import cv2
import numpy as np
import urllib.request
from pathlib import Path
from typing import List, Tuple, Optional
import os


# ---------------------------------------------------------------------------
# Rutas de modelos MediaPipe (se descargan automaticamente si no existen)
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent / "models"

FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

FACE_DETECTOR_MODEL_PATH = MODELS_DIR / "blaze_face_short_range.tflite"
FACE_LANDMARKER_MODEL_PATH = MODELS_DIR / "face_landmarker.task"


def download_model(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Descargando modelo: {dest.name} ...")
        urllib.request.urlretrieve(url, str(dest))
        print(f"Modelo descargado: {dest.name}")


# ---------------------------------------------------------------------------
# Indices de landmarks MediaPipe relevantes para alineacion
# El FaceLandmarker devuelve 478 puntos (malla facial densa)
# Usamos el centro de cada ojo para alinear
# ---------------------------------------------------------------------------

# Indices del contorno del ojo izquierdo y derecho en la malla de 478 puntos
LEFT_EYE_CENTER_IDX = 468  # iris izquierdo centro
RIGHT_EYE_CENTER_IDX = 473  # iris derecho centro

# Alternativas si no hay iris (modelo sin tracking de iris): esquinas del ojo
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263


# ---------------------------------------------------------------------------
# Detector principal: MediaPipe Tasks
# ---------------------------------------------------------------------------


class MediaPipeDetector:
    """
    Detector de rostros usando MediaPipe Tasks API (>= 0.10.x).
    Combina FaceDetector (bounding box) y FaceLandmarker (478 puntos).
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        import mediapipe as mp

        self.mp = mp
        self.min_confidence = min_detection_confidence

        download_model(FACE_DETECTOR_MODEL_URL, FACE_DETECTOR_MODEL_PATH)
        download_model(FACE_LANDMARKER_MODEL_URL, FACE_LANDMARKER_MODEL_PATH)

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Detector de bounding boxes
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        det_options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_DETECTOR_MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        )
        self.detector = FaceDetector.create_from_options(det_options)

        # Landmarker para alineacion
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        lm_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
        )
        self.landmarker = FaceLandmarker.create_from_options(lm_options)

    def _to_mp_image(self, bgr_image: np.ndarray):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Devuelve lista de detecciones con:
          - box: [x, y, w, h] en pixeles
          - confidence: float
          - landmarks: array (N, 2) en pixeles o None
        """
        h, w = image.shape[:2]
        mp_img = self._to_mp_image(image)

        det_result = self.detector.detect(mp_img)
        lm_result = self.landmarker.detect(mp_img)

        # Construir mapa de landmarks por posicion de bounding box
        lm_map = {}
        for face_lms in lm_result.face_landmarks:
            pts = np.array([[lm.x * w, lm.y * h] for lm in face_lms], dtype=np.float32)
            # Calcular centro del conjunto de landmarks
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            lm_map[(round(cx), round(cy))] = pts

        detections = []
        for det in det_result.detections:
            bb = det.bounding_box
            x, y, bw, bh = bb.origin_x, bb.origin_y, bb.width, bb.height
            score = det.categories[0].score

            # Buscar landmarks correspondientes (el mas cercano al centro del box)
            box_cx = x + bw / 2
            box_cy = y + bh / 2
            best_pts = None
            best_dist = float("inf")
            for (lcx, lcy), pts in lm_map.items():
                dist = (lcx - box_cx) ** 2 + (lcy - box_cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_pts = pts

            detections.append(
                {
                    "box": [x, y, bw, bh],
                    "confidence": score,
                    "landmarks": best_pts,
                }
            )

        return detections


# ---------------------------------------------------------------------------
# Fallback: OpenCV (solo si MediaPipe no esta disponible)
# ---------------------------------------------------------------------------


class OpenCVDetector:
    """Detector fallback con Haar Cascade. Menos preciso."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError("No se pudo cargar haarcascade_frontalface_alt2.xml")

    def detect(self, image: np.ndarray) -> List[dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(50, 50)
        )
        results = []
        for x, y, w, h in faces:
            results.append(
                {
                    "box": [int(x), int(y), int(w), int(h)],
                    "confidence": 0.8,
                    "landmarks": None,
                }
            )
        return results


# ---------------------------------------------------------------------------
# Alineacion facial
# ---------------------------------------------------------------------------


def align_face(
    image: np.ndarray,
    landmarks: Optional[np.ndarray],
    output_size: int = 224,
) -> Optional[np.ndarray]:
    """
    Alinea el rostro usando los centros de los ojos como referencia.
    Si no hay landmarks, hace un resize simple.

    Args:
        image: recorte del rostro (ya con padding aplicado)
        landmarks: array (N, 2) de landmarks en coordenadas del recorte
        output_size: tamano cuadrado de salida

    Returns:
        Imagen alineada de tamano (output_size, output_size, 3)
    """
    if landmarks is None or len(landmarks) < 478:
        return cv2.resize(
            image, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4
        )

    h, w = image.shape[:2]

    # Obtener centros de los ojos
    # Intentar iris primero (indices 468 y 473), luego esquinas
    if len(landmarks) >= 478:
        left_eye = landmarks[LEFT_EYE_CENTER_IDX]
        right_eye = landmarks[RIGHT_EYE_CENTER_IDX]
    else:
        left_eye = landmarks[[LEFT_EYE_INNER, LEFT_EYE_OUTER]].mean(axis=0)
        right_eye = landmarks[[RIGHT_EYE_INNER, RIGHT_EYE_OUTER]].mean(axis=0)

    # Angulo entre ojos
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Centro entre los ojos
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    # Distancia deseada entre ojos en la imagen de salida
    desired_eye_y = 0.35  # ojos al 35% desde arriba
    desired_left_eye_x = 0.30
    desired_dist = output_size * (1.0 - 2 * desired_left_eye_x)

    current_dist = np.sqrt(dx * dx + dy * dy)
    scale = desired_dist / (current_dist + 1e-6)

    # Matriz de rotacion y escala centrada en el punto medio de los ojos
    M = cv2.getRotationMatrix2D(eye_center, angle, scale)

    # Ajustar traslacion para que los ojos queden en la posicion deseada
    tX = output_size * 0.5
    tY = output_size * desired_eye_y
    M[0, 2] += tX - eye_center[0]
    M[1, 2] += tY - eye_center[1]

    aligned = cv2.warpAffine(
        image,
        M,
        (output_size, output_size),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


# ---------------------------------------------------------------------------
# FaceExtractor: clase principal
# ---------------------------------------------------------------------------


class FaceExtractor:
    """
    Extrae rostros de videos e imagenes con alta calidad.

    Flujo por deteccion:
        1. Deteccion del bounding box (MediaPipe BlazeFace)
        2. Expansion con padding proporcional
        3. Verificacion: el recorte no toca bordes de la imagen
        4. Alineacion con landmarks (rotacion por angulo entre ojos)
        5. Verificacion de calidad (nitidez, brillo, contraste)
        6. Guardado en output_size x output_size
    """

    def __init__(
        self,
        detector_type: str = "mediapipe",
        output_size: int = 224,
        min_confidence: float = 0.5,
        padding: float = 0.3,
        min_face_size: int = 60,
        quality_check: bool = True,
        min_sharpness: float = 80.0,
    ):
        """
        Args:
            detector_type: "mediapipe" (recomendado) u "opencv" (fallback)
            output_size: tamano cuadrado de salida en pixeles
            min_confidence: confianza minima del detector
            padding: fraccion de padding alrededor del box (0.3 = 30%)
            min_face_size: tamano minimo del box detectado en pixeles
            quality_check: si True, descarta imagenes borrosas/oscuras
            min_sharpness: varianza del Laplaciano minima para aceptar
        """
        self.output_size = output_size
        self.min_confidence = min_confidence
        self.padding = padding
        self.min_face_size = min_face_size
        self.quality_check = quality_check
        self.min_sharpness = min_sharpness

        if detector_type == "mediapipe":
            try:
                self.detector = MediaPipeDetector(
                    min_detection_confidence=min_confidence
                )
                print("Detector: MediaPipe Tasks (BlazeFace + FaceLandmarker)")
            except Exception as e:
                print(f"MediaPipe no disponible ({e}), usando OpenCV como fallback")
                self.detector = OpenCVDetector()
        elif detector_type == "opencv":
            self.detector = OpenCVDetector()
            print("Detector: OpenCV Haar Cascade (fallback)")
        else:
            raise ValueError(
                f"detector_type debe ser 'mediapipe' u 'opencv', no '{detector_type}'"
            )

    def _crop_with_padding(
        self, image: np.ndarray, box: List[int]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Recorta el rostro con padding. Retorna (recorte, landmarks_en_recorte, valido).
        Si el recorte se sale de la imagen, retorna (None, None, False).
        """
        x, y, w, h = box
        img_h, img_w = image.shape[:2]

        if w < self.min_face_size or h < self.min_face_size:
            return None, None, False

        # Padding proporcional al mayor lado
        side = max(w, h)
        pad = int(side * self.padding)

        x1 = x - pad
        y1 = y - pad
        x2 = x + w + pad
        y2 = y + h + pad

        # Descartar si el recorte se sale significativamente de la imagen
        # Tolerancia del 5% del lado
        tol = int(side * 0.05)
        if x1 < -tol or y1 < -tol or x2 > img_w + tol or y2 > img_h + tol:
            return None, None, False

        # Clipear a los bordes de la imagen
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(img_w, x2)
        y2c = min(img_h, y2)

        crop = image[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            return None, None, False

        return crop, (x1c, y1c), True

    def _check_quality(self, face: np.ndarray) -> bool:
        if not self.quality_check:
            return True
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        contrast = np.std(gray)
        return (
            sharpness >= self.min_sharpness and 30 < brightness < 230 and contrast > 20
        )

    def process_frame(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Detecta y extrae todos los rostros validos de un frame.

        Returns:
            Lista de imagenes de rostros alineadas, tamano output_size x output_size
        """
        detections = self.detector.detect(frame)
        results = []

        for det in detections:
            if det["confidence"] < self.min_confidence:
                continue

            crop, crop_origin, valid = self._crop_with_padding(frame, det["box"])
            if not valid or crop is None:
                continue

            # Ajustar landmarks al sistema de coordenadas del recorte
            aligned_lms = None
            if det["landmarks"] is not None:
                ox, oy = crop_origin
                lms_in_crop = det["landmarks"] - np.array([ox, oy], dtype=np.float32)
                aligned_lms = lms_in_crop

            face = align_face(crop, aligned_lms, self.output_size)
            if face is None:
                continue

            if not self._check_quality(face):
                continue

            results.append(face)

        return results

    def process_image(self, image_path: str) -> List[np.ndarray]:
        """
        Procesa una imagen estatica y retorna los rostros extraidos.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"No se pudo leer: {image_path}")
            return []
        return self.process_frame(image)

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        sample_interval: float = 0.5,
        max_faces_per_video: int = 100,
    ) -> int:
        """
        Procesa un video, extrae rostros y los guarda.

        Args:
            video_path: ruta al video
            output_dir: directorio de salida
            sample_interval: segundos entre frames analizados
            max_faces_per_video: limite de rostros por video

        Returns:
            Numero de rostros guardados
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_name = Path(video_path).stem

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: no se puede abrir {video_path}")
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = max(1, int(fps * sample_interval))

        print(f"Procesando: {video_name}")
        print(
            f"  FPS: {fps:.1f} | Frames: {total_frames} | Intervalo: cada {sample_rate} frames"
        )

        face_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue

            faces = self.process_frame(frame)

            for face in faces:
                if face_count >= max_faces_per_video:
                    break
                filename = f"{video_name}_f{frame_idx:06d}_r{face_count:04d}.jpg"
                cv2.imwrite(
                    str(output_dir / filename),
                    face,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                face_count += 1

            if face_count >= max_faces_per_video:
                print(f"  Limite alcanzado: {max_faces_per_video} rostros")
                break

            frame_idx += 1

        cap.release()
        print(f"  Rostros extraidos: {face_count}")
        return face_count
