import cv2
import numpy as np
import mediapipe as mp
import math
import time
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Try to import Windows sound library (optional feature)
try:
    import winsound
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


# ------------------------------
# Configuration settings
# ------------------------------
class Config:
    WIDTH, HEIGHT = 1280, 720
    PINCH_THRESHOLD = 40
    SMOOTHING = 0.6
    BRUSH_SIZE = 8
    NEON_GLOW = True
    HUD_COLOR = (255, 255, 0)
    ARC_CENTER = (640, 0)
    ARC_RADIUS = 150
    ARC_THICKNESS = 60


# ------------------------------
# Sound engine
# ------------------------------
class SoundEngine:
    def __init__(self):
        self.active = False
        self.velocity = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop)
        self.thread.daemon = True
        self.thread.start()

    def set_drawing(self, is_drawing, velocity):
        self.active = is_drawing
        self.velocity = velocity

    def _loop(self):
        while not self.stop_event.is_set():
            if AUDIO_AVAILABLE and self.active:
                try:
                    freq = int(200 + (self.velocity * 5))
                    freq = max(100, min(freq, 800))
                    winsound.Beep(freq, 40)
                except:
                    pass
            else:
                time.sleep(0.05)


# ------------------------------
# Hand tracking (updated for Mediapipe 0.10.x)
# ------------------------------
class HandSystem:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.prev_pos = (0, 0)

    def process(self, img):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        result = self.detector.detect(mp_image)

        if result.hand_landmarks:
            h, w, c = img.shape
            points = []
            for lm in result.hand_landmarks[0]:
                points.append((int(lm.x * w), int(lm.y * h)))
            return points
        return None

    def draw_sci_fi_hud(self, img, points, pinch_dist):
        if not points:
            return img
        overlay = img.copy()
        # Keep your HUD drawing logic unchanged
        return cv2.addWeighted(overlay, 0.7, img, 0.3, 0)


# ------------------------------
# Arc color palette UI
# ------------------------------
class ArcPalette:
    def __init__(self):
        self.colors = [
            ((0, 0, 255), "RED"),
            ((0, 165, 255), "ORANGE"),
            ((0, 255, 255), "YELLOW"),
            ((0, 255, 0), "GREEN"),
            ((255, 255, 0), "CYAN"),
            ((255, 0, 255), "PURPLE"),
            ((255, 255, 255), "WHITE"),
            ((0, 0, 0), "CLEAR")
        ]
        self.selected_index = 4

    def draw(self, img, hover_pt):
        # Keep your palette drawing logic unchanged
        return -1


# ------------------------------
# Main application loop
# ------------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, Config.WIDTH)
    cap.set(4, Config.HEIGHT)

    hand_sys = HandSystem()
    palette = ArcPalette()
    sound = SoundEngine()

    canvas = np.zeros((Config.HEIGHT, Config.WIDTH, 3), dtype=np.uint8)
    smooth_x, smooth_y = 0, 0
    current_color = (255, 255, 0)

    print("IRON CANVAS ACTIVATED")

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        points = hand_sys.process(img)

        # Keep your drawing logic unchanged

        cv2.imshow("Iron Canvas Pro", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    sound.stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

