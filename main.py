import os
import cv2
import numpy as np


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
DIR = "Results"


def get_file_name(prefix):
    return DIR + "\\" + prefix + "_" + FILE_NAME


if not os.path.exists(DIR):
    os.mkdir(DIR)


# OPEN AND NORMALIZE
image = cv2.imread(FILE_NAME, 0)
normalized_image = (cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)) * 255
cv2.imwrite(get_file_name("1norm"), normalized_image)

normalized_image = cv2.imread(get_file_name("1norm"), 0)

# FILTERING
RADIUS = 10
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RADIUS, RADIUS))
opening = cv2.morphologyEx(normalized_image, cv2.MORPH_OPEN, kernel)
cv2.imwrite(get_file_name("2opening"), opening)

# SEGMENTATION
ret3, thresh_image = cv2.threshold(opening, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite(get_file_name("3threshold"), thresh_image)

filled_image = FILL(thresh_image)
cv2.imwrite(get_file_name("4filled"), filled_image)


# EXTRACTION
contours, bin = cv2.findContours(filled_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)


display_image = cv2.cvtColor(normalized_image.copy(), cv2.COLOR_GRAY2BGR)


GREEN = 200
cv2.drawContours(display_image, contours, -1, (0, GREEN, 0), 6, cv2.LINE_4)

cv2.imshow("Result", display_image)

cv2.imwrite(get_file_name("5result"), display_image)
cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
