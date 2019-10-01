import math
from sklearn import neighbors
import os
import pickle
import face_recognition
import cv2
import numpy as np
import time
import pandas as pd

col_names = ['video', 'frame', 'date', 'time', 'emp_id', 'emp_name', 'loc_x1', 'loc_x2', 'loc_y1', 'loc_y2']
df = pd.DataFrame(columns = col_names)

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to testing folder
PATH_TO_TESTING = os.path.join(CWD_PATH, 'testing')

# Load face encodings
with open('trained_knn_model.clf', 'rb') as f:
  knn_clf = pickle.load(f)
print('Model loaded.')

# Open the input movie file
input_movie_name = "test3.mp4"
input_movie_path = os.path.join(PATH_TO_TESTING, input_movie_name)
input_movie = cv2.VideoCapture(input_movie_path)
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
timestr = time.strftime("%d-%m-%Y_%H-%M_")
output_name = timestr + input_movie_name
output_movie_name = output_name + '_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
output_movie = cv2.VideoWriter(output_movie_name, fourcc, 30.012158, (1920, 1080))

frame_number = 0
distance_threshold = 0.6
predictions = []

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1
    print("Frame", frame_number)

    if frame_number % 2 == 0:
        
    
        # Quit when the input video file ends
        if not ret:
            print('No video.')
            break

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = small_frame[:, :, ::-1]


        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)

        # If no faces are found in the image, return an empty result.
        if len(face_locations) == 0:
            predictions = []
            print('No faces found.')
            #continue

        else:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=60)

            # Use the KNN model to find the best matches for the test face
            closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

            # Predict classes and remove classifications that aren't within the threshold
            predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(face_encodings), face_locations, are_matches)]

            datestr = time.strftime("%d-%m-%Y")
            timestr = time.strftime("%H-%M-%S")
            

    # Label the results
    for name, (top, right, bottom, left) in predictions:
        if not name:
            continue

        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        scaleback = 2
        top *= scaleback
        top -= 30
        right *= scaleback
        bottom *= scaleback
        left *= scaleback

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, top - 35), (right, top), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 0.6, (255, 255, 255), 1)

        df.loc[frame_number] = (input_movie_name, frame_number,
                                datestr, timestr,
                                name+'_id', name,
                                left,right,top,bottom)
            
    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

df.to_csv(output_name + '_outputLogs.csv', index = False)
print('Excel done!')
# All done!
input_movie.release()
output_movie.release()
cv2.destroyAllWindows()
