import cv2
import numpy as np
from matplotlib import pyplot as plt


def FILL(input_image):
    image_floodfill = input_image.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    height, width = input_image.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(image_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    image_floodfill_inv = cv2.bitwise_not(image_floodfill)

    # Combine the two images to get the foreground.
    output_image = input_image | image_floodfill_inv
    return output_image




FILE_NAME = '1m129070954eff0224p2933m2f1.png'

# OPEN AND NORMALIZE
img = cv2.imread(FILE_NAME, 0)
# norm = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16S)


# FILTERING
RADIUS = 10
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RADIUS, RADIUS))
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


# SEGMENTATION
# grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret3,th3 = cv2.threshold(opening, 120, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

filled_image = FILL(th3)
cv2.imwrite("filled_"+FILE_NAME, filled_image)


# EXTRACTION
contours, bin = cv2.findContours(filled_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

normalized_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)



display_image = cv2.cvtColor(normalized_image.copy(), cv2.COLOR_GRAY2BGR)

color = 120
cv2.drawContours(display_image, contours, -1, (0, color, 0), 6, cv2.LINE_4)

cv2.imshow("Foreground", display_image)

display_image = display_image * 255
cv2.imwrite("result_"+FILE_NAME, display_image)
cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
