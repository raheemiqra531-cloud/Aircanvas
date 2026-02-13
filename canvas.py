import cv2
import numpy as np
import mediapipe as mp
import math
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import pytesseract
import time

# ---------------------------- CONFIG ----------------------------
WIDTH, HEIGHT = 1280, 720
PINCH_THRESHOLD = 40  # Not used anymore, one-finger drawing
SMOOTHING = 0.1       # Reduced for faster responsiveness
MAX_JUMP_DISTANCE = 100  # Allow more distance between frames

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ---------------------------- HAND TRACKER ----------------------------
class HandTracker:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            lm = results.multi_hand_landmarks[0].landmark
            return [(int(p.x * w), int(p.y * h)) for p in lm]
        return None

# ---------------------------- SHAPE RECOGNITION ----------------------------
def recognize_shape(stroke):
    if len(stroke) < 30:
        return None

    pts = np.array(stroke)
    peri = cv2.arcLength(pts, False)
    approx = cv2.approxPolyDP(pts, 0.02 * peri, False)

    if len(approx) == 2:
        return "LINE"
    elif len(approx) == 4:
        return "RECTANGLE"
    elif len(approx) > 6:
        return "CIRCLE"
    return None

# ---------------------------- MODERN SMART BOARD ----------------------------
class AirSmartBoard(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("AI Air Smart Board PRO")
        self.geometry("1500x850")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, WIDTH)
        self.cap.set(4, HEIGHT)

        self.tracker = HandTracker()
        self.canvas_layer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        self.tool = "BRUSH"
        self.color = (0, 255, 255)
        self.brush_size = 5

        self.prev_point = None
        self.smooth_x = None
        self.smooth_y = None
        self.stroke_points = []

        self.history = []
        self.redo_stack = []

        self.show_help = True

        # ---------------- Layout ----------------
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.grid(row=0, column=1, sticky="nsew")

        # Sidebar Buttons
        ctk.CTkLabel(self.sidebar, text="TOOLS", font=("Arial", 18, "bold")).pack(pady=10)
        ctk.CTkButton(self.sidebar, text="Brush", command=lambda: self.set_tool("BRUSH")).pack(pady=5)
        ctk.CTkButton(self.sidebar, text="Eraser", command=lambda: self.set_tool("ERASER")).pack(pady=5)
        ctk.CTkButton(self.sidebar, text="Undo", command=self.undo).pack(pady=5)
        ctk.CTkButton(self.sidebar, text="Redo", command=self.redo).pack(pady=5)
        ctk.CTkButton(self.sidebar, text="Save PNG", command=self.save_image).pack(pady=5)
        ctk.CTkButton(self.sidebar, text="OCR Text", command=self.recognize_text).pack(pady=5)
        ctk.CTkButton(self.sidebar, text="Toggle Help", command=self.toggle_help).pack(pady=5)

        # Color Buttons
        ctk.CTkLabel(self.sidebar, text="COLORS", font=("Arial", 18, "bold")).pack(pady=10)
        colors = {
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Blue": (255, 0, 0),
            "Yellow": (0, 255, 255),
            "White": (255, 255, 255),
            "Cyan": (255, 255, 0),
        }
        for name, bgr in colors.items():
            ctk.CTkButton(self.sidebar, text=name, command=lambda c=bgr: self.set_color(c)).pack(pady=3)

        self.update_video()

    # ---------------------------- CONTROLS ----------------------------
    def set_tool(self, tool):
        self.tool = tool

    def set_color(self, color):
        self.tool = "BRUSH"
        self.color = color

    def toggle_help(self):
        self.show_help = not self.show_help

    def undo(self):
        if self.history:
            self.redo_stack.append(self.canvas_layer.copy())
            self.canvas_layer = self.history.pop()

    def redo(self):
        if self.redo_stack:
            self.canvas_layer = self.redo_stack.pop()

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png")
        if file_path:
            cv2.imwrite(file_path, self.canvas_layer)

    def recognize_text(self):
        gray = cv2.cvtColor(self.canvas_layer, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        print("Recognized Text:\n", text)

    # ---------------------------- MAIN LOOP ----------------------------
    def update_video(self):
        start = time.time()
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)

        points = self.tracker.process(frame)

        if points:
            # Index finger tip for drawing
            current = points[8]

            # Smoothing
            if self.smooth_x is None:
                self.smooth_x, self.smooth_y = current
            self.smooth_x = int(SMOOTHING * current[0] + (1 - SMOOTHING) * self.smooth_x)
            self.smooth_y = int(SMOOTHING * current[1] + (1 - SMOOTHING) * self.smooth_y)
            current_point = (self.smooth_x, self.smooth_y)

            # Draw if index finger is up (simple)
            self.stroke_points.append(current_point)

            if self.prev_point is not None:
                dx = current_point[0] - self.prev_point[0]
                dy = current_point[1] - self.prev_point[1]
                distance = math.hypot(dx, dy)

                if distance < MAX_JUMP_DISTANCE * 2:  # allow faster motion
                    steps = max(1, int(distance / 2))
                    for i in range(steps):
                        ix = int(self.prev_point[0] + dx * i / steps)
                        iy = int(self.prev_point[1] + dy * i / steps)
                        cv2.circle(
                            self.canvas_layer,
                            (ix, iy),
                            self.brush_size if self.tool == "BRUSH" else self.brush_size * 4,
                            self.color if self.tool == "BRUSH" else (0, 0, 0),
                            -1
                        )

            self.prev_point = current_point
        else:
            self.prev_point = None
            self.stroke_points = []

        combined = cv2.add(frame, self.canvas_layer)

        # Feature guide
        if self.show_help:
            lines = [
                "AI AIR SMART BOARD PRO",
                "",
                "Use Index Finger to Draw",
                "Brush / Eraser via Sidebar",
                "Undo / Redo",
                "Save PNG",
                "OCR Text",
                f"Current Tool: {self.tool}"
            ]
            y = 60
            for line in lines:
                cv2.putText(combined, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y += 30

        # FPS
        fps = int(1 / (time.time() - start + 0.0001))
        cv2.putText(combined, f"FPS: {fps}", (1100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk

        self.after(10, self.update_video)

# ---------------------------- RUN ----------------------------
if __name__ == "__main__":
    app = AirSmartBoard()
    app.mainloop()


