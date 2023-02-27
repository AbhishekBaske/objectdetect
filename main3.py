import cv2
import numpy as np

# Load the image
img = cv2.imread('44.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection to find edges
edges = cv2.Canny(gray, 100, 200)

# Apply morphological closing to close small gaps between edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Find contours in the closed image
contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours to find the ones that meet our criteria
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 5000: # adjust this threshold as needed
        # Draw the contour on the original image
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

# Display the result
cv2.imshow('Roof Surfaces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
