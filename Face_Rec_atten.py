import cv2
import numpy as np
import face_recognition as face_rec
import os
from datetime import datetime
import time

def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimensions = (width, height)
    return cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)

path = 'sample_images'
sampleimg = []
samplename = []
myList = os.listdir(path)

#get rid of .jpg extension
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    sampleimg.append(curImg)
    samplename.append(os.path.splitext(cl)[0])

def findEncoding(images):
    encoding_list = []
    for img in images:
        img = resize(img, 0.5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        encoding_list.append(encodeimg)
    return encoding_list

recorded_names = {}  # Dictionary to store already recorded names

def MarkAttendance(name):
    global recorded_names  # Access the global recorded_names dictionary.

    today = datetime.now().strftime('%Y-%m-%d')
    folder_path = 'attendence_folder' #specify folder path
    csv_filename = f'{folder_path}/attendance_{today}.csv'

    if today not in recorded_names:
        recorded_names[today] = set()

    if name not in recorded_names[today]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(csv_filename, 'a') as f:
            if today not in recorded_names:
                f.write('Name, Date, Time\n')  # Write header if the file is new
            now = datetime.now()
            datestr = now.strftime('%Y-%m-%d')
            timestr = now.strftime('%H:%M')
            f.write(f'{name}, {datestr}, {timestr}\n')
            recorded_names[today].add(name)

encode_list = findEncoding(sampleimg)

vid = cv2.VideoCapture(0)

while True:

    success, frame = vid.read()
    frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    frames = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces_in_frame = face_rec.face_locations(frames)
    encode_in_frame = face_rec.face_encodings(frames, faces_in_frame)

    for encodeFace, faceloc in zip(encode_in_frame, faces_in_frame):
        matches = face_rec.compare_faces(encode_list, encodeFace)
        facedis = face_rec.face_distance(encode_list, encodeFace)
        print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name = samplename[matchIndex].upper()
            y1, x2, y2, x1 = faceloc

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            MarkAttendance(name)

    cv2.imshow('video', frame)
    cv2.waitKey(1)
