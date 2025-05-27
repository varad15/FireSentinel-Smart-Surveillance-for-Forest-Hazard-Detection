import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import *
from tkinter import messagebox, filedialog
from tkinter.ttk import Combobox
from ultralytics import YOLO
import cv2
import json
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np

class FireDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Fire Detection App")

        self.label = Label(master, text="Fire Detection App", font=("Helvetica", 20))
        self.label.pack(pady=10)

        self.video_source_label = Label(master, text="Select Video Source:", font=("Helvetica", 12))
        self.video_source_label.pack()

        self.video_source_combo = Combobox(master, values=["Camera", "Video File"])
        self.video_source_combo.pack(pady=5)
        self.video_source_combo.current(0)

        self.start_button = Button(master, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=5)

        self.stop_button = Button(master, text="Stop Detection", command=self.stop_detection, state=DISABLED)
        self.stop_button.pack(pady=5)

        self.accuracy_label = Label(master, text="", font=("Helvetica", 16))
        self.accuracy_label.pack(pady=10)

        self.model = YOLO('best.pt')
        self.cap = None
        self.is_detecting = False
        self.total_frames = 0
        self.fire_detections = 0
        self.no_fire_frames = 0
        self.detections = []

        self.true_labels = []  # Ground truth labels (1 for fire, 0 for no fire)
        self.pred_labels = []  # Predictions (1 for fire, 0 for no fire)

    def start_detection(self):
        video_source = self.video_source_combo.get()
        if video_source == "Camera":
            self.cap = cv2.VideoCapture(0)  # Open webcam
        else:
            filename = filedialog.askopenfilename()
            if not filename:
                messagebox.showerror("Error", "Please select a video file.")
                return
            self.cap = cv2.VideoCapture(filename)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open video source.")
            return

        self.is_detecting = True
        self.start_button.config(state=DISABLED)
        self.stop_button.config(state=NORMAL)
        self.detect_fire()

    def stop_detection(self):
        self.is_detecting = False
        self.start_button.config(state=NORMAL)
        self.stop_button.config(state=DISABLED)
        self.evaluate_accuracy()

    def detect_fire(self):
        if self.is_detecting:
            ret, frame = self.cap.read()
            if not ret:
                self.is_detecting = False
                self.start_button.config(state=NORMAL)
                self.stop_button.config(state=DISABLED)
                messagebox.showinfo("Info", "Video source ended.")
                self.evaluate_accuracy()
                return

            self.total_frames += 1

            frame = cv2.resize(frame, (640, 480))
            result = self.model(frame, stream=True)

            fire_detected_in_frame = False
            for info in result:
                boxes = info.boxes
                for box in boxes:
                    class_id = box.cls[0]
                    if class_id != 0:  # Check if the detected class is 'fire' (assuming fire class is 0)
                        continue  # Skip this detection
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    self.fire_detections += 1
                    fire_detected_in_frame = True
                    self.detections.append({
                        "image_id": self.total_frames,
                        "category_id": 0,  # assuming fire class is always 0
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": 1.0  # setting confidence to 1.0
                    })
                    cv2.putText(frame, 'FIRE DETECTED', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Append prediction labels (1 for fire, 0 for no fire)
            self.pred_labels.append(1 if fire_detected_in_frame else 0)

            # For now, ground truth is assumed (for demonstration purposes)
            # In real scenarios, this should be replaced with real ground truth data.
            self.true_labels.append(1 if fire_detected_in_frame else 0)

            total_processed_frames = self.total_frames + self.no_fire_frames  # Include frames where no fire is detected
            accuracy = (self.fire_detections / total_processed_frames) * 100 if total_processed_frames > 0 else 0
            self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}% Fire Detections: {self.fire_detections} / {total_processed_frames}")

            cv2.imshow('Fire Detection', frame)
            cv2.waitKey(1)

            if self.is_detecting:
                self.master.after(10, self.detect_fire)

    def evaluate_accuracy(self):
        if len(self.true_labels) == len(self.pred_labels) and len(self.true_labels) > 0:
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(self.true_labels, self.pred_labels)
            print("Confusion Matrix:")
            print(conf_matrix)

            # Display confusion matrix using seaborn
            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.show()

            # Calculate accuracy, precision, recall, F1-score
            print("Classification Report:")
            print(classification_report(self.true_labels, self.pred_labels))

            # Calculate accuracy
            accuracy = accuracy_score(self.true_labels, self.pred_labels) * 100
            print(f"Accuracy: {accuracy:.2f}%")

            # Save detections to file
            with open('detections.json', 'w') as f:
                json.dump(self.detections, f)
        else:
            print("Mismatch in prediction and ground truth data length.")

    def __del__(self):
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    root = Tk()
    root.geometry("800x600")
    app = FireDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
