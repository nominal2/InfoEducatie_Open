import cv2

webcam = cv2.VideoCapture(0)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detect = True
id_count = 0
while True:

    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    if len(face_coordinates) == 0:
        detect = False
    else:
        detect = True


    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)


    cv2.imshow('Name_Your_Frame', frame)
    key = cv2.waitKey(1)

    if 0xFF == ord('q'):
        break


webcam.release()