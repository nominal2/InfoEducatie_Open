import cv2
import numpy as np

class EuclideanTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, centroids):
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(centroid)
        else:
            input_ids = list(self.objects.keys())
            object_ids = list(self.objects.keys())
            distance_matrix = np.zeros((len(input_ids), len(centroids)))

            for i, object_id in enumerate(object_ids):
                for j, centroid in enumerate(centroids):
                    dx = centroid[0] - self.objects[object_id][0]
                    dy = centroid[1] - self.objects[object_id][1]
                    distance_matrix[i, j] = np.sqrt(dx*dx + dy*dy)

            rows = distance_matrix.min(axis=1).argsort()
            cols = distance_matrix.argmin(axis=1)

            used_rows = set()
            used_cols = set()

            for row in rows:
                if row in used_rows:
                    continue

                object_id = object_ids[row]
                min_distance = distance_matrix[row, cols[row]]
                if min_distance < self.max_distance:
                    self.objects[object_id] = centroids[cols[row]]
                    self.disappeared[object_id] = 0
                    used_rows.add(row)
                    used_cols.add(cols[row])

            unused_rows = set(range(0, distance_matrix.shape[0])).difference(used_rows)
            unused_cols = set(range(0, distance_matrix.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = input_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(centroids[col])

        return self.objects


euclidean_tracker = EuclideanTracker(max_disappeared=50, max_distance=50)
cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    persons = cascade_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    centroids = []
    for (x, y, w, h) in persons:
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        centroids.append((center_x, center_y))

    tracked_objects = euclidean_tracker.update(centroids)

    for object_id, centroid in tracked_objects.items():
        x, y = centroid
        cv2.putText(frame, f"ID: {object_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    # Display the frame with detected persons and their IDs (optional)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close any open windows
cap.release()
cv2.destroyAllWindows()
