import cv2 as cv

cam = cv.VideoCapture(2)

cv.namedWindow("test")
img_counter = 0

running = True
while running:
    try:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv.imshow("test", frame)
        
        k = cv.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        # elif k%256 == 32:
        #     # SPACE pressed
        #     img_name = "opencv_frame_{}.png".format(img_counter)
        #     cv.imwrite(img_name, frame)
        #     print("{} written!".format(img_name))
        #     img_counter += 1
    except KeyboardInterrupt:
        running = False

cam.release()
# cv.waitKey(0)
cv.destroyAllWindows()