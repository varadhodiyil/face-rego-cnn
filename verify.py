
import cv2
from model import CNNBuilder
from opencv_utils import OpenCVHelper


open_cv_helper = OpenCVHelper()
_cnn = CNNBuilder()
cap = cv2.VideoCapture(0)
_cnn.load_model()

while cap.isOpened():

    ret, image = cap.read()
    

    img_face, pos_face = open_cv_helper.convert_img(image)

    if img_face is not None:
        cv2.rectangle(image, (pos_face[0], pos_face[2]),
                      (pos_face[1], pos_face[3]), (255, 0, 0), 5)
        img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
        user, acc = _cnn.classify(img_face)

        cv2.putText(image, "%s Accuracy %f " % (user, acc),
                    (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Faces found", image)

    else:
        print("No Detection!")

    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
