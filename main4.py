import cv2
import numpy as np

# Read the image
img = cv2.imread("44.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply edge detection to the image
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply Hough Line Transform to detect lines in the image
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

# Iterate over the lines and draw them on the original image
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Show the image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
