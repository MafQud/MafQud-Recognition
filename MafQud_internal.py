import os
import cv2
import numpy as np
from time import time
from mtcnn import MTCNN
from deepface.detectors.FaceDetector import alignment_procedure


from keras_facenet import FaceNet
class MafQud_internal:
    def __init__(self):
        self.detector = MTCNN()
        self.facenet = FaceNet().model

    def detect_align_face(self, path):
        """Detect face with 4 landmarks and align 

        Args:
            path (str): path for image

        Returns:
            numpy.ndarray, list: image array, list of face landmarks and bounding box data
        """
        t0 = time()
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(img)
        if len(detections) == 0 or len(detections) > 1:
            return None

        # Crop and align the face
        x, y, w, h = detections[0]["box"]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        face = img[y1:y2, x1:x2]
        keypoints = detections[0]["keypoints"]
        face_align = alignment_procedure(
            face, keypoints["left_eye"], keypoints["right_eye"]
        )

        # resize the face and convert to numpy array
        face_align = cv2.resize(face_align, (160, 160))
        face_array = np.asarray(face_align)
        t1 = time() - t0
        print(f"Face  detected and aligned in {t1}")
        return face_array

    def load_faces(self, path):
        """Load faces in all images for a given person

        Args:
            path (str): path for the given person

        Returns:
            list, list : list of faces in all the person images, list of face landmarks and bounding box data
        """
        t0 = time()
        faces = list()
        for filename in os.listdir(path):
            dir_ = path + filename
            face = self.detect_align_face(dir_)
            if face is None:
                print("No face found.")
                continue
            faces.append(face)

        t1 = time() - t0
        print(f"All person images, Face detectes and aligned in {t1}")
        return faces

    def load_dataset(self, path):
        """Load faces in all images for a all people
        with their ids.

        Args:
            path (str): path for dataset
            data_dir_name (str): file name for csv file
        Returns:
            numpy.ndarray, numpy.ndarray: array of  all faces for a all people, id fo each face
        """
        t0 = time()
        X, y = list(), list()
        for subdir in os.listdir(path):
            dir_ = path + subdir + "/"
            if not os.path.isdir(dir_):
                print("Directory doesn't exist")
                continue
            faces = self.load_faces(dir_)
            labels = [subdir for _ in range(len(faces))]
            print(">loaded %d examples for class: %s" % (len(faces), subdir))
            X.extend(faces)
            y.extend(labels)

        t1 = time() - t0
        print(f"All people images, Face detected and aligned in {t1}")
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int)


    def feature_encoding(self, img):
        """Extract 128 numerical feature from face array

        Args:
            model (tensorflow.python.keras.engine.functional.Functional): acenet keras model
            img (numpy.ndarray): face array

        Returns:
            numpy.ndarray: array of 128 numerica feature
        """
        t0 = time()
        img = img.astype("float32")
        mean, std = img.mean(), img.std()
        img = (img - mean) / std
        samples = np.expand_dims(img, axis=0)
        yhat = self.facenet.predict(samples)
        t1 = time() - t0
        print(f"Face encoding extracted in {t1}")
        return yhat[0]

    def get_encodings(self, X):
        """Extract 128 numerical feature all face arrays for each person

        Args:
            model (tensorflow.python.keras.engine.functional.Functional): acenet keras model
            X (numpy.ndarray): array of all faces for each person
    
        Returns:
            numpy.ndarray: array of encoding for each face
        """
        t0 = time()
        newX = list()
        print("Encoding started")
        for img in X:
            embedding = self.feature_encoding(img)
            newX.append(embedding)
            print("Extraxting features for face No. %d" % (len(newX)))

        t1 = time() - t0
        print(f"dataset face encodings extracted in {t1}")
        return np.asarray(newX)

