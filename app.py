from collections import defaultdict
from deepface import DeepFace
import threading
import chromadb
import time
import uuid
import cv2
import os


class FaceAuthentication:
    def __init__(self, source=0, db="facedb", root="faces"):
        self.cap = cv2.VideoCapture(source)
  
        if not os.path.exists(root):
            os.makedirs(root)

        self.root = root

        self.client = chromadb.PersistentClient(root)
        self.db = self.client.get_or_create_collection(
            name=db,
            metadata={
                "hnsw:space": 'cosine',
            },
        )

        self.cropped_frame = None
        self.identity = None
        

    def detect(self, frame, threshold=0.5):
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
        except: pass

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
                if self.cropped_frame is None or self.cropped_frame.size == 0:

                    print("Face not detected")
                else: 
                    embedding = DeepFace.represent(img_path=self.cropped_frame)[0]
                    print(len(embedding['embedding']))
      
                    idx = str(uuid.uuid4())
                    filename = idx + '.jpg'

                    path = os.path.join(self.root, '__faces__', full_name)
                    if not os.path.exists(path):
                        os.makedirs(path)

                    cv2.imwrite(os.path.join(path, filename), self.cropped_frame)

                    self.db.add(
                        ids=[idx],
                        embeddings=[embedding['embedding']],
                        metadatas=[{'name': full_name}]
                    )
                    print("Successfully registered")
                break
            
            try:
                cv2.rectangle(*boundary)
            except: pass

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


    def retrieve(self,folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files[:5]:
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
        
                self.cropped_frame, boundary = self.detect(img, 0.8)
            
                if self.cropped_frame is not None and self.cropped_frame.size != 0:
                    try:
                        embedding = DeepFace.represent(img_path=self.cropped_frame)[0]
                        print(len(embedding['embedding']))
                    except: pass
                    else:
                        
                        full_name = os.path.normpath(root).split(os.sep)[-1]
                        print(full_name)
                        
                        idx = str(uuid.uuid4())
                        filename = idx + '.jpg'

                        path = os.path.join(self.root, '__faces__', full_name)
                        if not os.path.exists(path):
                            os.makedirs(path)

                        cv2.imwrite(os.path.join(path, filename), self.cropped_frame)

                        self.db.add(
                            ids=[idx],
                            embeddings=[embedding['embedding']],
                            metadatas=[{'name': full_name}]
                        )

        print("Successfully registered")


    def run(self):
        window_name = "Face verification"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        dfs = threading.Thread(target=self.verify, daemon=True)
        dfs.start()
  
        while True:
            ret, frame = self.cap.read()

            self.cropped_frame, boundary = self.detect(frame)
            
            try:
                if self.identity is not None:
                    FONT = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, self.identity, boundary[1], FONT, 1, boundary[3], boundary[4])

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
        previous_time = time.time()
        
        while True:
            current_time = time.time()

            if current_time - previous_time > 0.5:
                previous_time = current_time

                try:
                    embedding = DeepFace.represent(img_path=self.cropped_frame)[0]
                    print(len(embedding['embedding']))
                    results = self.db.query(
                        query_embeddings=[embedding['embedding']],  # Replace with your unknown embeddings
                        n_results=1
                    )
                    #print(results['metadatas'], results['distances'])
                    print(results['distances'][0][0])
                    """
                    names = [metadata['name'] for metadata in results['metadatas'][0]]
                    distances = [distance for distance in results['distances'][0]]
                    #print(names, distances)

                    # Create a dictionary to store the total sum and count for each name
                    name_stats = {}

                    # Loop through the names and values
                    for i, name_dict in enumerate(names):
                        name = name_dict
                        distance = distances[i]
                        
                        # If the name is already in the dictionary, update the sum and count
                        if name in name_stats:
                            name_stats[name]['sum'] += distance
                            name_stats[name]['count'] += 1
                        else:
                            # If the name is not in the dictionary, initialize the sum and count
                            name_stats[name] = {'sum': distance, 'count': 1}

                    # Calculate the average for each name
                    #print(name_stats)
                    averages = {name: stats['sum'] * stats['count'] / len(name_stats) for name, stats in name_stats.items()}

                    # Print the result
                    print(averages)

                    lowest_name = min(averages, key=averages.get)
                    lowest_score = averages[lowest_name]
                    print(lowest_name, lowest_score)

                    if lowest_score < 0.4:
                        self.identity = lowest_name
                    else:
                        self.identity = lowest_name
                    """
                    if results['distances'][0][0] < 0.45:
                        print(results['metadatas'][0][0]['name'])
                        self.identity = results['metadatas'][0][0]['name']
                    else:
                        self.identity = None

                except: pass 

if __name__ == '__main__':
    app = FaceAuthentication()
    app.register(5)
    #app.retrieve('C:\Workspace\deepface\Celebrity Faces Dataset')
    app.run()