import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import time

model_path = ".\optimized_mobilenet_lstm_model.h5"
optimized_model = tf.keras.models.load_model(model_path)
class_labels = ["NonViolence", "Violence"]
video_capture = None
is_inference_running = False

def open_video_file():
    global video_capture, is_inference_running
    file_path = filedialog.askopenfilename()
    if file_path:
        video_capture = cv2.VideoCapture(file_path)
        is_inference_running = True
        run_inference()

def run_inference():
    global video_capture, is_inference_running
    while is_inference_running:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (64, 64))
        frame = frame / 255.0
        start_time = time.time()
        frame = np.expand_dims(frame, axis=0).astype(object)  
        predictions = optimized_model.predict(frame)
        end_time = time.time()
        inference_time = end_time - start_time

        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
        result_text = f"Class: {predicted_class_label} | Confidence: {confidence:.2f} | Inference Time: {inference_time:.2f} seconds"

        cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Fighting Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_inference_running = False
            break

    if video_capture:
        video_capture.release()
    cv2.destroyAllWindows()

def stop_inference():
    global is_inference_running
    is_inference_running = False

root = tk.Tk()
root.title("Real-time Human Fighting Detection")
open_button = ttk.Button(root, text="Open Video File", command=open_video_file)
open_button.pack()
start_button = ttk.Button(root, text="Start Inference", command=run_inference)
start_button.pack()
quit_button = ttk.Button(root, text="Quit Inference", command=stop_inference)
quit_button.pack()

root.mainloop()