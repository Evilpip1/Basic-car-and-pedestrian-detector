from cv2 import cv2
import numpy as np

video = cv2.VideoCapture("video.mp4")
car_classifier = cv2.CascadeClassifier("cars.xml")
people_classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")

while True:
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_classifier.detectMultiScale(gray, 1.3, 3)
    people = people_classifier.detectMultiScale(gray, 1.1, 2)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    for (x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 3)

    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
