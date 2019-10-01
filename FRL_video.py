import face_recognition
import cv2
import numpy as np
import os
import pickle
import time

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to training data folder
PATH_TO_TRAINING = os.path.join(CWD_PATH, 'data')

# Path to testing folder
PATH_TO_TESTING = os.path.join(CWD_PATH, 'testing')

# Load face encodings
with open('dataset_known_faces.dat', 'rb') as dkf:
  known_face_encodings = pickle.load(dkf)
print('Model loaded.')

# Grab the list of names and the list of encodings
known_face_names = list(known_face_encodings.keys())
known_face_encodings = np.array(list(known_face_encodings.values()))

# Open the input movie file
input_movie_name = "ambuj4.mp4"
input_movie = cv2.VideoCapture(input_movie_name)
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
timestr = time.strftime("%d-%m-%Y_%H-%M")
output_movie_name = timestr + input_movie_name + 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
output_movie = cv2.VideoWriter(output_movie_name, fourcc, 30.012158, (1920, 1080))

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
output_movie.release()
cv2.destroyAllWindows()
