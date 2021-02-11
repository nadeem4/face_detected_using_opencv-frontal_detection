import cv2


trained_data = cv2.CascadeClassifier("haarcascade_frontal_default.xml")

# img = cv2.imread("RDJ.jpg")

webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read, frame = webcam.read()
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_data.detectMultiScale(grayscale_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 100), 2)

    cv2.imshow("Face detector using open cv", frame)
    cv2.waitKey(1)


webcam.release()
"""
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_data.detectMultiScale(grayscale_img)

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 100), 2)

cv2.imshow("Face detector using open cv", img)

cv2.waitKey()
"""
