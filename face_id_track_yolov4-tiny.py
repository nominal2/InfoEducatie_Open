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

    def update(self, frame, net, output_layers):
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        centroids = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    centroids.append((center_x, center_y))

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


net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
output_layers = net.getUnconnectedOutLayersNames()

euclidean_tracker = EuclideanTracker(max_disappeared=50, max_distance=50)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    tracked_objects = euclidean_tracker.update(frame, net, output_layers)

    for object_id, centroid in tracked_objects.items():
        x, y = centroid
        cv2.putText(frame, f"ID: {object_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
