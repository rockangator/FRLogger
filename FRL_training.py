import face_recognition
import os
import pickle
import time

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to training data folder
PATH_TO_TRAINING = os.path.join(CWD_PATH, 'data')

#Total number of files in training.
total_files = 0
for root, dirs, files in os.walk(PATH_TO_TRAINING):
  total_files += len(files)
	
# Load known faces and learn how to recognize them.
print("Loading faces to train...")
filex = 1
known_face_encodings = {}

for person in os.listdir(PATH_TO_TRAINING):
  path_to_person = os.path.join(PATH_TO_TRAINING, person)
  print('\nKnown face:', person)
  
  for filename in os.listdir(path_to_person):
    print('{}/{} - {}'.format(filex,total_files,filename))
    path_to_file = os.path.join(path_to_person, filename)

    person_image = face_recognition.load_image_file(path_to_file)    
    person_face_encodings = face_recognition.face_encodings(person_image, num_jitters=100)

    if len(person_face_encodings) > 0:
      person_face_encoding = person_face_encodings[0]
      known_face_encodings[person] = person_face_encoding
    else:
      print("Skipped, no face found.")      

    filex+=1
    
print("\nFaces loaded.")

#saving to pickle
timestr = time.strftime("%d-%m-%Y_%H-%M")
pickle_name = timestr + '_' + 'MUGSdataset_known_faces.dat'
with open(pickle_name, 'wb') as dkf:
  pickle.dump(known_face_encodings, dkf)

print("Pickle saved.")
