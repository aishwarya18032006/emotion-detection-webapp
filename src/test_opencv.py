import cv2

# Print OpenCV version
print("OpenCV version:", cv2.__version__)

# Try to read an image (you can use any image on your computer)
img = cv2.imread("C:/Users/raish/Desktop/dnn/sample.jpg")

# Check if image was loaded
if img is None:
    print("Error: Image not found. Please check the path.")
else:
    cv2.imshow("Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
