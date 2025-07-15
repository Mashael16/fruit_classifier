# from tensorflow.keras.models import load_model
#  # TensorFlow is required for Keras to work
# import cv2  # Install opencv-python
# import numpy as np
#
# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)
#
# # Load the model
# model = load_model("C:\\Users\\Msali\\OneDrive\\سطح المكتب\\train summer\\secTaskAi\\fruit_classifier\\keras_model.h5", compile=False)
#
# # Load the labels
# class_names = open("C:\\Users\\Msali\\OneDrive\\سطح المكتب\\train summer\\secTaskAi\\fruit_classifier\\labels.txt", "r").readlines()
#
# # CAMERA can be 0 or 1 based on default camera of your computer
# camera = cv2.VideoCapture(0)
#
# while True:
#     # Grab the webcamera's image.
#     ret, image = camera.read()
#
#     # Resize the raw image into (224-height,224-width) pixels
#     image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
#
#     # Show the image in a window
#     cv2.imshow("Webcam Image", image)
#
#     # Make the image a numpy array and reshape it to the models input shape.
#     image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
#
#     # Normalize the image array
#     image = (image / 127.5) - 1
#
#     # Predicts the model
#     prediction = model.predict(image)
#     index = np.argmax(prediction)
#     class_name = class_names[index]
#     confidence_score = prediction[0][index]
#
#     # Print prediction and confidence score
#     print("Class:", class_name[2:], end="")
#     print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
#
#     # Listen to the keyboard for presses.
#     keyboard_input = cv2.waitKey(1)
#
#     # 27 is the ASCII for the esc key on your keyboard.
#     if keyboard_input == 27:
#         break
#
# camera.release()
# cv2.destroyAllWindows()
# import tkinter as tk
# from PIL import Image, ImageTk
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
#
# # تحميل النموذج والملصقات
# model = load_model("keras_model.h5", compile=False)
# class_names = open("labels.txt", "r").readlines()
#
# # إعداد نافذة GUI
# window = tk.Tk()
# window.title("تصنيف الفواكه")
# window.geometry("700x600")
#
# # لعرض الفيديو
# video_label = tk.Label(window)
# video_label.pack()
#
# # لعرض النتائج
# result_label = tk.Label(window, text="", font=("Arial", 20))
# result_label.pack(pady=20)
#
# # فتح الكاميرا
# cap = cv2.VideoCapture(0)
#
# def update_frame():
#     ret, frame = cap.read()
#     if not ret:
#         window.after(10, update_frame)
#         return
#
#     # عرض الفيديو
#     display_frame = cv2.resize(frame, (640, 480))
#     rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(rgb_frame)
#     imgtk = ImageTk.PhotoImage(image=img)
#     video_label.imgtk = imgtk
#     video_label.configure(image=imgtk)
#
#     # معالجة الصورة للتنبؤ
#     image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
#     image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
#     image = (image / 127.5) - 1  # Normalization
#
#     # تنبؤ النموذج
#     prediction = model.predict(image)
#     index = np.argmax(prediction)
#     class_name = class_names[index].strip()
#     confidence_score = prediction[0][index] * 100
#
#     # تحديث التسمية
#     result_text = f"الصنف: {class_name} \nالدقة: {confidence_score:.2f}%"
#     result_label.config(text=result_text)
#
#     # التحديث كل 100 مللي ثانية
#     window.after(100, update_frame)
#
# # بدء العرض
# update_frame()
# window.mainloop()
#
# # عند إغلاق النافذة يتم إغلاق الكاميرا
# cap.release()
# cv2.destroyAllWindows()
# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#
# import tkinter as tk
# from PIL import Image, ImageTk
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
#
# # تحميل النموذج
# model = load_model(r"C:\sectask\keras_model.h5", compile=False)
#
# class_names = open(r"C:\sectask\labels.txt", "r").readlines()
#
# # إعداد نافذة GUI
# window = tk.Tk()
# window.title("تصنيف الفواكه")
# window.geometry("700x600")
#
# # لعرض الفيديو
# video_label = tk.Label(window)
# video_label.pack()
#
# # لعرض النتائج
# result_label = tk.Label(window, text="", font=("Arial", 20))
# result_label.pack(pady=20)
#
# # فتح الكاميرا
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("⚠ تعذر فتح الكاميرا")
#
# def update_frame():
#     ret, frame = cap.read()
#     if not ret:
#         window.after(10, update_frame)
#         return
#
#     # عرض الفيديو
#     display_frame = cv2.resize(frame, (640, 480))
#     rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(rgb_frame)
#     imgtk = ImageTk.PhotoImage(image=img)
#     video_label.imgtk = imgtk
#     video_label.configure(image=imgtk)
#
#     # معالجة الصورة للتنبؤ
#     image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
#     image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
#     image = (image / 127.5) - 1  # Normalization
#
#     # تنبؤ النموذج
#     prediction = model.predict(image)
#     index = np.argmax(prediction)
#     class_name = class_names[index].strip()
#     confidence_score = prediction[0][index] * 100
#
#     # تحديث التسمية
#     result_text = f"الصنف: {class_name} \nالدقة: {confidence_score:.2f}%"
#     result_label.config(text=result_text)
#
#     # التحديث كل 100 مللي ثانية
#     window.after(100, update_frame)
#
# # بدء العرض
# update_frame()
# window.mainloop()
#
# # عند الإغلاق
# cap.release()
# cv2.destroyAllWindows()
# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#
# import tkinter as tk
# from tkinter import messagebox
# from PIL import Image, ImageTk
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
#
# # Load model safely
# try:
#     model = load_model(r"C:\sectask\keras_model.h5", compile=False)
# except Exception as e:
#     messagebox.showerror("Error", f"Failed to load model: {e}")
#     exit()
#
# # Load class names cleanly
# with open(r"C:\sectask\labels.txt", "r") as f:
#     class_names = [line.strip() for line in f if line.strip()]
#
# window = tk.Tk()
# window.title("Fruit Classifier")
# window.geometry("700x600")
#
# video_label = tk.Label(window)
# video_label.pack()
#
# result_label = tk.Label(window, text="", font=("Arial", 20))
# result_label.pack(pady=20)
#
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     messagebox.showerror("Error", "Cannot open camera")
#     exit()
#
# confidence_threshold = 50
#
# def update_frame():
#     ret, frame = cap.read()
#     if not ret:
#         window.after(10, update_frame)
#         return
#
#     display_frame = cv2.resize(frame, (640, 480))
#     rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(rgb_frame)
#     imgtk = ImageTk.PhotoImage(image=img)
#     video_label.imgtk = imgtk
#     video_label.configure(image=imgtk)
#
#     image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
#     image = image.astype(np.float32)
#     image = (image / 127.5) - 1
#     image = np.expand_dims(image, axis=0)
#
#     prediction = model.predict(image)
#     index = np.argmax(prediction)
#     confidence_score = prediction[0][index] * 100
#
#     if confidence_score > confidence_threshold:
#         class_name = class_names[index]
#         result_text = f"Class: {class_name}\nConfidence: {confidence_score:.2f}%"
#     else:
#         result_text = "Cannot confidently identify the fruit."
#
#     result_label.config(text=result_text)
#
#     window.after(100, update_frame)
#
# window.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), window.destroy()))
# update_frame()
# window.mainloop()
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model(r"C:\sectask\keras_model.h5", compile=False)

# Read labels and remove numbers
class_names = [line.strip().split(" ", 1)[-1] for line in open(r"C:\sectask\labels.txt", "r")]

# Create GUI window
window = tk.Tk()
window.title("Fruit Classifier")
window.geometry("700x600")

# Video display
video_label = tk.Label(window)
video_label.pack()

# Prediction results
result_label = tk.Label(window, text="", font=("Arial", 18), justify="left")
result_label.pack(pady=20)

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠ Cannot open camera")

def update_frame():
    ret, frame = cap.read()
    if not ret:
        window.after(10, update_frame)
        return

    # Show camera frame
    display_frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Prepare the frame for prediction
    image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    # Predict
    predictions = model.predict(image)[0]
    top_indices = predictions.argsort()[-3:][::-1]

    result_text = "Top Predictions:\n"
    for i in top_indices:
        class_label = class_names[i]
        score = predictions[i] * 100
        result_text += f"{class_label}: {score:.2f}%\n"

    result_label.config(text=result_text)

    # Repeat after 100ms
    window.after(100, update_frame)

# Start updating frames
update_frame()

# Graceful close
def on_close():
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_close)
window.mainloop()
