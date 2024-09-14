from deepface import DeepFace
from pathlib import Path
import pandas as pd
import numpy as np
import threading
import sqlite3
import glob
import time
import uuid
import cv2
import os

class FaceAuthentication:
  def __init__(self, source=0, db="face_database.db", root="db"):
    self.cap = cv2.VideoCapture(source)

    if not os.path.exists(root):
      os.makedirs(root)
    self.root = root
    self.db = db

    connection = sqlite3.connect(db)
    cursor = connection.cursor()
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        dir TEXT 
      )
    """)
    connection.commit()
    connection.close()

    self.cropped_frame = None
    self.identity = None


  def detect(self, frame=None, threshold=0):
    cropped_frame = None
    boundary = None
    try:
      face = DeepFace.extract_faces(img_path=frame)

      if face[0]['confidence'] >= threshold:
        face_data = face[0]['facial_area']
        x, y, w, h, leye, reye = face_data.values()

        cropped_frame = frame[y:y+h, x:x+w]
        
        COLOR = (0, 255, 0)
        THICKNESS = 2
        START_POINT = (x, y)
        END_POINT = (x + w, y + h)

        boundary = (frame, START_POINT, END_POINT, COLOR, THICKNESS)
    except:
      pass

    return cropped_frame, boundary


  def register(self, timeout=3):
    full_name = input("Enter your full name: ")
    start_time = time.time() 

    window_name = "Face registration"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    while True:
      elapsed_time = time.time() - start_time
      ret, frame = self.cap.read()

      self.cropped_frame, boundary = self.detect(frame)

      print(elapsed_time, timeout)
      if elapsed_time >= timeout:
        if self.cropped_frame is None and self.cropped_frame.size != 0:
          print("Face not detected")
        else:
          connection = sqlite3.connect(self.db)
          cursor = connection.cursor()
          cursor.execute('SELECT dir FROM users WHERE full_name = ? LIMIT 1', (full_name,))
          dir = cursor.fetchone()
          if not dir:
            dir = (os.path.join(self.root, str(uuid.uuid4())),)

            cursor.execute('INSERT INTO users (full_name, dir) VALUES (?, ?)', (full_name, dir[0]))
            connection.commit()

          if not os.path.exists(dir[0]):
            os.makedirs(dir[0])

          filename = str(uuid.uuid4()) + '.jpg'
          cv2.imwrite(os.path.join(dir[0], filename), self.cropped_frame)
          print("Successfully registered")

          connection.close()
        break
      
      try:
        cv2.rectangle(*boundary)
      except: pass
      cv2.imshow(window_name, frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.destroyAllWindows()


  def retrieve(self, folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
          img_path = os.path.join(root, file)
          img = cv2.imread(img_path)
       
          self.cropped_frame, boundary = self.detect(img, 0.9)
         
          if self.cropped_frame is not None and self.cropped_frame.size != 0:

            full_name = os.path.normpath(root).split(os.sep)[-1]

            connection = sqlite3.connect(self.db)
            cursor = connection.cursor()
            cursor.execute('SELECT dir FROM users WHERE full_name = ? LIMIT 1', (full_name,))
            dir = cursor.fetchone()
            if not dir:
              dir = (os.path.join(self.root, str(uuid.uuid4())),)
              
              cursor.execute('INSERT INTO users (full_name, dir) VALUES (?, ?)', (full_name, dir[0]))
              connection.commit()

            if not os.path.exists(dir[0]):
              os.makedirs(dir[0])

            filename = str(uuid.uuid4()) + '.jpg'
            cv2.imwrite(os.path.join(dir[0], filename), self.cropped_frame)


  def run(self):
    window_name = "Face verification"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    dfs = threading.Thread(target=self.verify, daemon=True)
    dfs.start()

    while True:
      ret, frame = self.cap.read()

      self.cropped_frame, boundary = self.detect(frame)
      if self.identity is not None:
        pass

      try:
        cv2.rectangle(*boundary)
      except: pass
      cv2.imshow(window_name, frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    try:
      cv2.rectangle(*boundary)
    except: pass
    cv2.destroyAllWindows()


  def verify(self):
    while True:
      self.identity = None
      try:
        dfs = DeepFace.find(self.cropped_frame, self.root, enforce_detection=False)
        print(dfs)
      except:
        pass    

if __name__ == '__main__':
  auth = FaceAuthentication()
  #auth.register(5)
  #auth.retrieve("C:\Workspace\deepface\Celebrity Faces Dataset")
  auth.run()