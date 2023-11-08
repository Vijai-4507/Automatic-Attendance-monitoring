import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)


known_faces = []
known_faces_names = []

jobs_image = face_recognition.load_image_file("photos/jobs.jpg")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]
known_faces.append(jobs_encoding)
known_faces_names.append("jobs")


students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    current_time = datetime.now().strftime("%H-%M-%S")

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        face_distances = face_recognition.face_distance(known_faces, face_encoding)

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_faces_names[best_match_index]
            face_names.append(name)

            if name in students:
                students.remove(name)
                print(students)

    for name in face_names:
        # Write to the CSV file with name and timestamp
        with open(f"{name}.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, current_time])

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
