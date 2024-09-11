from deepface import DeepFace
from pathlib import Path
import pandas as pd
import numpy as np
import threading
import time
import uuid
import cv2
import os

class FaceAuthentication:
  def __init__(self, source=0, db_path='db'):
    self.cap = cv2.VideoCapture(source)
    if not os.path.exists(db_path):
      os.makedirs(db_path)
    self.db_path = db_path

    self.image = None
    self.label = None

  def register(self, timeout=3):
    name = input("Enter your name: ")
    start_time = time.time()

    window_name = "Face registration"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    while True:
      elapsed_time = time.time() - start_time
      ret, frame = self.cap.read()

      face = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
      face_data = face[0]['facial_area']
      x, y, w, h, leye, reye = face_data.values()

      COLOR = (0, 0, 255)
      THICKNESS = 2
      START_POINT = (x, y)
      END_POINT = (x + w, y + h)
      
      print(elapsed_time, timeout)
      if elapsed_time >= timeout and face[0]['confidence'] > 0:
        if face[0]['confidence'] < 0.5:
          print("Face not detected")
          break
        else:
          id = uuid.uuid4()
          path = os.path.join(self.db_path, f"{str(id)}_{name}.jpg")
          cv2.imwrite(path, frame[y:y+h, x:x+w])

          print("Successfully registered")
          break

      cv2.rectangle(frame, START_POINT, END_POINT, COLOR, THICKNESS)
      cv2.imshow(window_name, frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.destroyAllWindows()


  def run(self):
    window_name = "Face verification"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    dfs = threading.Thread(target=self.verify, daemon=True)
    dfs.start()

    while True:
      ret, frame = self.cap.read()

      face = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
      face_data = face[0]['facial_area']
      x, y, w, h, leye, reye = face_data.values()

      COLOR = (0, 0, 255)
      THICKNESS = 2
      START_POINT = (x, y)
      END_POINT = (x + w, y + h)
      FONT = cv2.FONT_HERSHEY_SIMPLEX
      
      if face[0]['confidence'] >= 0.5:
        self.image = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, START_POINT, END_POINT, COLOR, THICKNESS)

        if self.label is not None:
          cv2.putText(frame, self.label, START_POINT, FONT, 1, COLOR, THICKNESS)
      else:
        self.image = None

      cv2.imshow(window_name, frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.destroyAllWindows()


  def verify(self):
    while True:
      if self.image is not None:
        dfs = DeepFace.find(img_path=self.image, db_path=self.db_path, enforce_detection=False, silent=True)
        if not dfs[0].empty:   
          filename = Path(dfs[0].iloc[0].identity).stem
          self.label = filename.split('_')[1]
      time.sleep(1)


if __name__ == '__main__':
  auth = FaceAuthentication()
  auth.register(5)
  auth.run()
  #dfs = DeepFace.verify("C:/Workspace/deepface/temp/ok.jpg", "C:/Workspace/deepface/db/66cc55a7-77c4-43b1-a76d-3ee1eac2d246_b.jpg", enforce_detection=False)
  #print(dfs)
