import time
from tkinter import messagebox
import customtkinter
from tkinter import *
import os
import cv2
import numpy as np
from PIL import ImageTk, Image
import datetime
import tkinter as tk
from tkinter import filedialog
from tkVideoPlayer import TkinterVideo
import imutils
from imutils.video import VideoStream


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1400x650")
        self.title("")

        self.resizable(width=True, height=True)
        self.file_path = None

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images with light and dark mode image
        image_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "test_images")
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(
            image_path, "icy.png")), size=(26, 26))


        self.home_image = customtkinter.CTkImage(
                                                 dark_image=Image.open(os.path.join(image_path, "image.png")), size=(20, 20))
        self.chat_image = customtkinter.CTkImage(
                                                 dark_image=Image.open(os.path.join(image_path, "video.png")), size=(20, 20))
        self.add_user_image = customtkinter.CTkImage(
                                                     dark_image=Image.open(os.path.join(image_path, "webcam.png")), size=(20, 20))

        # loading face detector model
        self.prototxtPath = os.path.sep.join(
            ["face_detector", "deploy.prototxt"])
        self.weightsPath = os.path.sep.join(
            ["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
        self.faceNet = cv2.dnn.readNet(self.prototxtPath, self.weightsPath)

        # loading age detector model
        self.prototxtPath = os.path.sep.join(
            ["age_detector", "age_deploy.prototxt"])
        self.weightsPath = os.path.sep.join(
            ["age_detector", "age_net.caffemodel"])
        self.ageNet = cv2.dnn.readNet(self.prototxtPath, self.weightsPath)
        self.AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
                            "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text=" Age Predictor", image=self.logo_image,
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Image",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                   image=self.home_image, anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.frame_2_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Video",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      image=self.chat_image, anchor="w", command=self.frame_2_button_event)
        self.frame_2_button.grid(row=2, column=0, sticky="ew")

        self.frame_3_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Real time",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      image=self.add_user_image, anchor="w", command=self.frame_3_button_event)
        self.frame_3_button.grid(row=3, column=0, sticky="ew")

        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(
            row=6, column=0, padx=20, pady=20, sticky="s")

# create home frame
        self.home_frame = customtkinter.CTkFrame(
            self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)

        self.panelB = None
        self.panelA = None
        self.image_frame = customtkinter.CTkFrame(
            self.home_frame, corner_radius=0.5, border_color="black")
        self.image_frame.pack(expand=True, fill="both", padx=40, pady=10)

        # frame for Buttons
        self.butframe = customtkinter.CTkFrame(
            self.home_frame, fg_color="transparent")
        self.butframe.pack(pady=30)
        self.home_frame_button = customtkinter.CTkButton(
            self.butframe, text="Open", command=self.open_img, height=40, width=100)
        self.home_frame_button.grid(row=0, column=0,  padx=5)

        self.home_frame_button1 = customtkinter.CTkButton(
            self.butframe, text="Predict", command=self.img_pred,  height=40, width=100)
        self.home_frame_button1.grid(row=0, column=1, padx=5)

        self.home_frame_button2 = customtkinter.CTkButton(
            self.butframe, text="Save", command=self.img_save, height=40, width=100)
        self.home_frame_button2.grid(row=0, column=2,  padx=5)

# create second frame
        self.second_frame = customtkinter.CTkFrame(
            self, corner_radius=0, fg_color="transparent")

        self.vid_player = tk.Label(
            self.second_frame, bg="black")
        self.vid_player.pack(expand=True, fill="both", padx=40, pady=10)

        self.btn_second_frame = customtkinter.CTkFrame(
            self.second_frame, corner_radius=0.5, fg_color="transparent")
        self.btn_second_frame.pack(pady=30)
        self.predict_vid_btn = customtkinter.CTkButton(
            self.btn_second_frame, text="Load video and Predict", command=self.handle_frame_vid, height=40, width=100)
        self.predict_vid_btn.pack(side="left", padx=5)
        self.play_pause_btn = customtkinter.CTkButton(
            self.btn_second_frame, text="Stop", command=self.pause, height=40, width=100)
        self.play_pause_btn.pack(side="left", padx=5)
        self.save_vid_btn = customtkinter.CTkButton(
            self.btn_second_frame, text="Save", command=self.save_vid, height=40, width=100)
        self.save_vid_btn.pack(side="left", padx=5)
        self.snap_btn_vid = customtkinter.CTkButton(
            self.btn_second_frame, text="Take Snapshot", command=self.takeSnapshot, height=40, width=100)
        self.snap_btn_vid.pack(expand=True, side="left", padx=5)
        self.vs = None


# create third frame
        self.third_frame = customtkinter.CTkFrame(
            self, corner_radius=0, fg_color="transparent")
        self.btn_third_frame = customtkinter.CTkFrame(
            self.third_frame, corner_radius=0.5, fg_color="transparent")
        self.btn_third_frame.pack(side="bottom", pady=30)

        self.live_label = tk.Label(self.third_frame, bg="black")
        self.live_label.pack(expand=True, fill=tk.BOTH, padx=40, pady=10)

        self.live_btn = customtkinter.CTkButton(
            self.btn_third_frame, text="Predict in Webcam", command=self.webcam_pred, height=40, width=100)
        self.live_btn.pack(expand=True, side="left", padx=5)

        self.stop_btn = customtkinter.CTkButton(
            self.btn_third_frame, text="Stop", command=self.cam_stop, height=40, width=100)
        self.stop_btn.pack(expand=True, side="left", padx=5)
        self.snap_btn = customtkinter.CTkButton(
            self.btn_third_frame, text="Take Snapshot", command=self.takeSnapshot, height=40, width=100)
        self.snap_btn.pack(expand=True, side="left", padx=5)
        self.cap = None
        self.select_frame_by_name("home")
    # frame api

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(
            fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.frame_2_button.configure(
            fg_color=("gray75", "gray25") if name == "frame_2" else "transparent")
        self.frame_3_button.configure(
            fg_color=("gray75", "gray25") if name == "frame_3" else "transparent")

        # show selected frame
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "frame_2":
            self.second_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.second_frame.grid_forget()
        if name == "frame_3":
            self.third_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.third_frame.grid_forget()

    def home_button_event(self):
        self.select_frame_by_name("home")

    def frame_2_button_event(self):
        self.select_frame_by_name("frame_2")

    def frame_3_button_event(self):
        self.select_frame_by_name("frame_3")

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    # image frame features
    def open_img(self):
        global x

        global count, eimg
        count = 0
        x = filedialog.askopenfilename()

        img = cv2.imread(x)
        img = cv2.resize(img, (600, 400))
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        eimg = img
        img = ImageTk.PhotoImage(img)

        if self.panelA is None:
            self.panelA = Label(self.image_frame, image=img)
            self.panelA.image = img
            self.panelA.grid(row=1, column=0, sticky="w", padx=10)
        else:
            self.panelA.configure(image=img)
            self.panelA.image = img
            self.panelB.configure(image="")

    def img_pred(self):
        try:
            global count, eimg
            count = 1
            image = cv2.imread(x)
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(
                image, 1.0, (300, 300), (104.0, 177.0, 123.0))

            self.faceNet.setInput(blob)
            detections = self.faceNet.forward()

            # Process the detections and display the result
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face = image[startY:endY, startX:endX]
                    faceBlob = cv2.dnn.blobFromImage(
                        face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

                    self.ageNet.setInput(faceBlob)
                    preds = self.ageNet.forward()

                    index = preds[0].argmax()
                    age = self.AGE_BUCKETS[index]
                    ageConfidence = preds[0][index]
                    text = "{}: {:.2f}%".format(age, ageConfidence * 100)
                    self.label_predict = customtkinter.CTkLabel(
                        self.home_frame, text="[INFO] Age between: {}, Accuracy {:.2f}%".format(age, ageConfidence * 100))
                    self.label_predict.pack(side="right", padx=10)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(image, (startX, startY),
                                  (endX, endY), (0, 0, 255), 2)
                    cv2.putText(image, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            image = cv2.resize(image, (600, 400))
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            eimg = image
            image = ImageTk.PhotoImage(image)
            self.panelB = Label(self.image_frame, image=image)
            self.panelB.image = image
            self.panelB.grid(row=1, column=1, sticky="e")
            self.panelB.configure(image=image)
            self.panelB.image = image
        except:
            messagebox.showerror("Error", "No photos to predict")

    def img_save(self):
        global count, eimg
        try:
            if count == 1 and eimg is not None:
                filename = filedialog.asksaveasfilename(
                    defaultextension=".jpg", filetypes=[("Image File", ".jpg")])
                eimg.save(filename)
                messagebox.showinfo(
                    "Success", "Image saved at: \n{}".format(filename))
        except:
            messagebox.showerror("Error", "No photos to save")

    # video frame features
    def save_vid(self):
        try:
            vs = cv2.VideoCapture(self.file_path)

            writer = None
            try:
                prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
                total = int(vs.get(prop))
                self.label_total_frames = customtkinter.CTkLabel(
                    self.second_frame, text="Total Frames: {} \nPlease wait for rendering...".format(total))
                self.label_total_frames.pack(side="right")

            except:
                messagebox.showinfo(
                    "[INFO] could not determine # of frames in video")
                messagebox.showinfo(
                    "[INFO] no approx. completion time can be provided")
                total = -1

            # lets begin our loop for video frames

            while (vs.isOpened()):

                (grabbed, frame) = vs.read()

                if not grabbed:
                    break

                frame = imutils.resize(frame, width=400)

                results = self.fame_pred(frame)

                for r in results:
                    text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)
                    (startX, startY, endX, endY) = r["loc"]

                    # lets put text and box on our image

                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY),
                                  (endX, endY), (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                if writer is None:
                    messagebox.showinfo(
                        "Location", "Please select location to save video")
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer_path = filedialog.asksaveasfilename(
                        defaultextension=".MJPG", filetypes=[("Video", ".MJPG")])
                    writer = cv2.VideoWriter(
                        writer_path, fourcc, 20, (frame.shape[1], frame.shape[0]), True)

                writer.write(frame)

            messagebox.showinfo(
                "Success", "Predicted video saved at:\n{}".format(writer_path))
        except:
            messagebox.showerror("Error", "No video to save")

    def pause(self):
        self.vs.release()
        self.vid_player.configure(text="Load video and Predict")

    def fame_pred(self, frame, minConf=0.5):
        results = []

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > minConf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                if face.shape[0] < 20 or face.shape[1] < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(
                    face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

                self.ageNet.setInput(faceBlob)
                preds = self.ageNet.forward()
                index = preds[0].argmax()
                age = self.AGE_BUCKETS[index]
                ageConfidence = preds[0][index]

                d = {
                    "loc": (startX, startY, endX, endY),
                    "age": (age, ageConfidence)
                }
                results.append(d)
        return results

    def handle_frame_vid(self):
        try:
            if self.vs is None:
                self.file_path = filedialog.askopenfilename()
                self.vs = cv2.VideoCapture(self.file_path)
                self.predict_vid_btn.configure(text="Video Predicting...")
                self.predict_vid_btn.configure(command=None)
            # lets begin our loop for video frames

            while (self.vs.isOpened()):

                (grabbed, frame) = self.vs.read()

                if not grabbed:
                    break

                frame = imutils.resize(frame, width=600)

                results = self.fame_pred(frame)

                for r in results:
                    text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)
                    (startX, startY, endX, endY) = r["loc"]

                    # lets put text and box on our image

                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY),
                                  (endX, endY), (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                self.snap = frame
                frame = cv2.resize(frame, (1000, 600))
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.vid_player.imgtk = imgtk
                self.vid_player.configure(image=imgtk)
                self.vid_player.after(10, self.handle_frame_vid)
                self.vid_player.wait_variable(self.cap, 10)
                if self.cap == 1:
                    break

            self.vs = None
            self.predict_vid_btn.configure(command=self.handle_frame_vid)
            self.predict_vid_btn.configure(text="Load video and Predict")
            messagebox.showinfo(" ", "Video ended")

        except:
            pass

    # real time frame features

    def webcam_pred(self):
        if self.cap is None:
            self.cap = VideoStream(src=0).start()
            time.sleep(2.0)
            self.live_btn.configure(text="Webcam Predicting...")
            self.live_btn.configure(command=None)
        try:
            while True:
                frame = self.cap.read()
                frame = imutils.resize(frame, width=800)

                results = self.fame_pred(frame)

                for r in results:
                    text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)
                    (startX, startY, endX, endY) = r["loc"]

                    # lets put text and box on our image

                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY),
                                  (endX, endY), (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                self.snap = frame
                frame = cv2.resize(frame, (1000, 600))
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.live_label.imgtk = imgtk
                self.live_label.configure(image=imgtk)
                self.live_label.after(10, self.webcam_pred)
                self.live_label.wait_variable(self.cap, 10)
                if self.cap.stop():
                    break

        except:
            pass

    def cam_stop(self):
        self.webcam_pred is False
        self.cap.stop()
        self.live_btn.configure(text="Reset", command=self.reset)
        self.live_label.configure(image="")

    def reset(self):
        self.webcam_pred is True
        self.cap = None

    def takeSnapshot(self):
        try:
            frame = self.snap

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            img_saved = filedialog.asksaveasfilename(defaultextension=".jpg", initialfile=ts, filetypes=[
                ("Image", ".jpg")])
            cv2.imwrite(img_saved, frame)
            messagebox.showinfo(
                "Success", "Capture saved successfully as: \n {}".format(img_saved))
        except:
            messagebox.showerror("Error", "No photos to save")


if __name__ == "__main__":
    app = App()
    app.mainloop()
