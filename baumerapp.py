import tkinter as tk
from tkinter import messagebox
import threading
import datetime
import pymongo
from PIL import Image, ImageTk
import numpy as np
import cv2
import neoapi
from paddleocr import PaddleOCR
from paddleocr.tools.infer.utility import draw_ocr

# === MongoDB Setup ===
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["ocr_database"]
collection = db["ocr_results"]

# === PaddleOCR Initialization ===
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# === Globals ===
baumer_cam = None
running = False
current_frame = None

# === GUI Setup ===
window = tk.Tk()
window.title("Baumer OCR Inspection Tool")
window.geometry("700x700")

canvas = tk.Canvas(window, width=480, height=360, bg="gray")
canvas.pack(pady=5)

label_status = tk.Label(window, text="Status: Idle")
label_status.pack(pady=2)

text_output = tk.Text(window, height=6, width=80, wrap=tk.WORD)
text_output.pack(pady=5)

# === Preprocessing ===
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(denoised, -1, kernel)
    thresh = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

def show_text_result(texts):
    text_output.delete("1.0", tk.END)
    if texts:
        text_output.insert(tk.END, "\n".join(texts))
    else:
        text_output.insert(tk.END, "[No text detected]")

def update_frame():
    global current_frame
    if not running or baumer_cam is None:
        return
    try:
        image = baumer_cam.GetImage()
        frame = image.GetNPArray()
        current_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img.resize((480, 360)))
        canvas.imgtk = imgtk
        canvas.create_image(0, 0, anchor='nw', image=imgtk)
    except Exception as e:
        print("Camera error:", e)
    if running:
        window.after(100, update_frame)

def start_camera():
    global baumer_cam, running
    try:
        baumer_cam = neoapi.Cam()
        baumer_cam.Connect()
        if baumer_cam.f.PixelFormat.GetEnumValueList().IsReadable('BGR8'):
            baumer_cam.f.PixelFormat.SetString('BGR8')
        else:
            baumer_cam.f.PixelFormat.SetString('Mono8')
        baumer_cam.f.ExposureTime.Set(10000)
        baumer_cam.f.AcquisitionFrameRateEnable.value = True
        baumer_cam.f.AcquisitionFrameRate.value = 10
        running = True
        label_status.config(text="Status: Camera started")
        update_frame()
    except Exception as e:
        messagebox.showerror("Camera Error", str(e))
        label_status.config(text="Status: Camera error")

def stop_camera():
    global running, baumer_cam
    running = False
    if baumer_cam:
        baumer_cam.Disconnect()
        baumer_cam = None
    canvas.delete("all")
    label_status.config(text="Status: Camera stopped")

def analyze_and_save():
    global current_frame
    if current_frame is None:
        messagebox.showerror("Error", "No frame captured yet.")
        return

    label_status.config(text="Status: Running OCR...")

    def worker():
        try:
            processed = preprocess_image(current_frame)
            result = ocr.ocr(processed, cls=True)
            texts = [line[1][0] for line in result[0]]
            show_text_result(texts)

            if texts:
                full_text = "\n".join(texts)
                collection.insert_one({
                    "timestamp": datetime.datetime.now(),
                    "detected_text": full_text
                })

                boxes = [line[0] for line in result[0]]
                txts = [line[1][0] for line in result[0]]
                scores = [line[1][1] for line in result[0]]

                font_path = "C:/Windows/Fonts/arial.ttf"
                image_with_boxes = draw_ocr(processed, boxes, txts, scores, font_path=font_path)
                output_img = Image.fromarray(image_with_boxes)
                output_img.save("ocr_result.jpg")

                label_status.config(text="Status: OCR completed and saved")
            else:
                label_status.config(text="Status: No text found")

        except Exception as e:
            messagebox.showerror("OCR Error", str(e))
            label_status.config(text="Status: OCR error")

    threading.Thread(target=worker, daemon=True).start()

# === Buttons ===
tk.Button(window, text="Start", width=30, command=start_camera).pack(pady=5)
tk.Button(window, text="Inspection", width=30, command=analyze_and_save).pack(pady=5)
tk.Button(window, text="Stop", width=30, command=stop_camera).pack(pady=5)

# === Exit Cleanup ===
window.protocol("WM_DELETE_WINDOW", lambda: (stop_camera(), window.destroy()))
window.mainloop()
