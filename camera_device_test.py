# test the camera decvice number 
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            print(f"Camera index {i} works.")
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        else:
            print(f"Camera index {i} opened but no frame.")
    else:
        print(f"Camera index {i} not available.")

    cap.release()