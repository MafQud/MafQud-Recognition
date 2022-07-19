from pathlib import Path
from typing import Dict, Optional
import cv2
import joblib
import pickle
import numpy as np
from mtcnn import MTCNN
from sklearn import neighbors
from deepface.detectors.FaceDetector import alignment_procedure
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
)
from urllib.request import Request, urlopen
from keras_facenet import FaceNet
from enum import Enum, auto
from time import sleep


class ModelState(Enum):
    RETRAINING = auto()
    CHECKING = auto()
    IDLE = auto()


class CaseType(Enum):
    LOST = auto()
    FOUND = auto()


class KNNModel:
    def __init__(self, model_path) -> None:
        self.knn_model = joblib.load(open(model_path, "rb"))
        self.state = ModelState.IDLE

    def change_state(self, state: ModelState):
        # TODO prevent transition for checking to retraining and vice versa
        if state != ModelState.IDLE and self.state != ModelState.IDLE:
            while (self.state != ModelState.IDLE):
                sleep(0.01)
        self.state = state


class MafQud:

    def __init__(self, lost, found) -> None:
        self.detector = MTCNN()
        self.facenet = FaceNet().model
        self.knn_lost_path = lost.path
        self.knn_lost = KNNModel(lost.path)
        self.lost_data = lost.data
        self.lost_labels = lost.labels
        self.knn_found_path = found.path
        self.found_data = found.data
        self.found_labels = found.labels
        self.knn_found = KNNModel(found.path)

    def detect_align_face(self, img: np.ndarray) -> np.ndarray:
        """Detect face with 4 landmarks and align

        Args:
            img (np.ndarray): image array

        Returns:
            numpy.ndarray, list: image array, list of face landmarks and bounding box data
        """
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

        return face_array

    def feature_encoding(self, img: np.ndarray) -> np.ndarray:
        """Extract 128 numerical feature from face array

        Args:
            img (numpy.ndarray): face array

        Returns:
            numpy.ndarray: array of 128 numerica feature
        """
        img = img.astype("float32")
        mean, std = img.mean(), img.std()
        img = (img - mean) / std
        samples = np.expand_dims(img, axis=0)
        yhat = self.facenet.predict(samples)

        return yhat[0]

    def encode_photo(self, url: str) -> np.ndarray:
        """Extract 128 numerical feature from face array

        Args:
            url (str): url to image

        Returns:
            numpy.ndarray: array of 128 numerical feature
        """
        req = Request(url, headers={"User-Agent": "XYZ/3.0"})
        data = urlopen(req, timeout=4000)
        arr = np.asarray(bytearray(data.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        face = self.detect_align_face(img)
        if face is None:
            return None

        embedding = self.feature_encoding(face).reshape(1, -1)

        return embedding

    def normalize_images_(self, x) -> np.ndarray:
        """Normalize training data

        Args:
            kind (str): type of normalization. Defaults to "l2".
        Returns:
            numpy.ndarray: X normalized
        """

        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        # in_encoder = Normalizer(norm=kind)
        # X = in_encoder.transform(self.data)
        return y

    def normalize_images(self):
        X = np.float32([self.normalize_images_(self.data[image])
                       for image in range(self.data.shape[0])])
        return X

    def KNN_Classifier(self, case_type: CaseType, n_neighbors: Optional[int] = 3):
        """KNN classifier to classify faces

        Args:
            n_neighbors (int, optional):number of nearest neighbors Defaults to 1.

        Returns:
            sklearn_model: KNN model
        """
        X = self.normalize_images()
        knn_clf = neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors, algorithm="ball_tree", weights="distance"
        )
        if case_type == CaseType.LOST:
            knn_clf.fit(X, self.lost_labels)
        elif case_type == CaseType.FOUND:
            knn_clf.fit(X, self.found_labels)

        return knn_clf

    def evaluate(self, X_test, y_test):
        """Evaluate model
        Args:
            model (tensorflow.python.keras.engine.functional.Functional): acenet keras model
            X_test (numpy.ndarray): array of all faces for each person
            y_test (numpy.ndarray): array of labels for each face

        Returns:
            accuracy (float): model accuracy
            precision (float): model precision
            recall (float): model recall
            f1 (float): model f1_score"""

        X_test = np.float32([self.normalize_images_(image)
                            for image in X_test])
        y_pred = self.knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(
            y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        print("as")
        print("Accuracy                                   : %.3f" % accuracy)
        print("Precision                                   : %.3f" % precision)
        print("Recall                                      : %.3f" % recall)
        print("F1-Score                                    : %.3f" % f1)
        print(
            "\nPrecision Recall F1-Score Support Per Class : \n",
            precision_recall_fscore_support(
                y_test, y_pred, average="weighted", zero_division=0
            ),
        )
        print("\nClassification Report                       : ")
        print(classification_report(y_test, y_pred, zero_division=0))
        return accuracy, precision, recall, f1

    def dump_model(self, case_type: CaseType):
        """Dump model to file
        """
        if case_type == CaseType.LOST:
            knn_path = self.knn_lost_path
            knn_model = self.knn_lost
        elif case_type == CaseType.FOUND:
            knn_path = self.knn_found_path
            knn_model = self.knn_found

        if knn_path is not None:
            with open(knn_path, "wb") as f:
                pickle.dump(knn_model, f)

    def check_face_identity(
        self,
        encodings: np.ndarray,
        case_type: CaseType,
        *,
        threshold: Optional[int] = 15,
        n_neighbors: Optional[int] = 9
    ) -> Dict[int, int]:
        """Check face identity

        Args:
            encodings (numpy.darray): face encodings
            threshold (int, optional): threshold for face identification Defaults to 0.5.

        Returns:
            identity (Dict[int, int]): return ids with number of images.
        """

        if case_type == CaseType.LOST:
            model_to_run = self.knn_found
            labels_to_look = self.found_labels
        elif case_type == CaseType.FOUND:
            model_to_run = self.knn_lost
            labels_to_look = self.lost_labels

        model_to_run.change_state(ModelState.CHECKING)

        closest_distances = model_to_run.kneighbors(
            encodings.reshape(1, -1), n_neighbors=n_neighbors
        )
        ids = {}
        distances = closest_distances[0][0]
        indices = closest_distances[1][0]
        for idx, distance in zip(indices, distances):
            #print(idx, self.labels[idx], distance)
            if distance <= threshold:
                if not ids.__contains__(labels_to_look[idx]):
                    ids[labels_to_look[idx]] = (1, distance)
                else:
                    num_photo = ids[labels_to_look[idx]][0] + 1
                    dist = ids[labels_to_look[idx]][1] + distance
                    ids[labels_to_look[idx]] = (num_photo, dist)

        if len(ids) == 0:
            return {-1: 0}
        else:
            ids_distances = [
                [],
                [],
                []
            ]

            for id, val in ids.items():
                ids_distances[val[0]-1].append((id, val[1]/val[0]))

            for id_distance in ids_distances:
                id_distance.sort(key=lambda i: i[1])

            ids_confidence = {}

            for num_photos in range(0, len(ids_distances)):
                for rank in range(0, len(ids_distances[num_photos])):
                    eqn = (num_photos+1)/3.0 - (rank+1)*0.03
                    ids_confidence[ids_distances[num_photos][rank][0]] = eqn

            model_to_run.change_state(ModelState.IDLE)

            return ids_confidence

    def retrain_model(self, new_encodings: np.ndarray, identity: int, case_type: CaseType):
        """Retrain model on new images

        Args:
            new_encodings (numpy.darray): new face encodings
            identity (int): identity of the face
        """
        if case_type == CaseType.LOST:
            self.knn_lost.change_state(ModelState.RETRAINING)
            for new_encoding in new_encodings:
                self.lost_data = np.vstack([self.lost_data, new_encoding])
                self.lost_labels = np.append(self.lost_labels, identity)
                self.knn_lost.knn_model = self.KNN_Classifier(case_type)

            self.dump_model(case_type)
            self.knn_lost.change_state(ModelState.IDLE)

        elif case_type == CaseType.FOUND:
            self.knn_found.change_state(ModelState.RETRAINING)
            for new_encoding in new_encodings:
                self.found_data = np.vstack([self.found_data, new_encoding])
                self.found_labels = np.append(self.found_labels, identity)
                self.knn_found.knn_model = self.KNN_Classifier(case_type)

            self.dump_model(case_type)
            self.knn_found.change_state(ModelState.IDLE)
