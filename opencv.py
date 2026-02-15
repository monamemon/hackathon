import cv2

# Read image
image = cv2.imread("image.jpeg")

# Display image
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Resize image
resized = cv2.resize(image, (300, 300))
cv2.imshow("Resized", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

blur = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Blur", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Edge detection
edges = cv2.Canny(image, 100, 200)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Draw a rectangle
cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), 2)

cv2.imshow("Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
