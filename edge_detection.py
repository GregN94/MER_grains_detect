import os
import cv2
import numpy as np

GREEN = (0, 200, 0)
BLUE = (200, 0, 0)
YELLOW = (51, 230, 230)
RED = (0, 0, 200)

DIR = "EdgeDetection"


def GetAllImages():
    return [dir for dir in os.listdir(".") if dir.endswith(".png")]


def GetFileName(prefix, image_name):
    return DIR + "\\" + image_name.split(".")[0] + "\\" + prefix + "_" + image_name


def CreateResultDirectory(images):
    if not os.path.exists(DIR):
        os.mkdir(DIR)

    for image in images:
        new_dir_name = DIR + "\\" + image.split(".")[0]
        if not os.path.exists(new_dir_name):
            os.mkdir(new_dir_name)


def IsBorderContour(contour, imageSize):

    (width, height) = imageSize
    x, y, w, h = cv2.boundingRect(contour)
    minX = 0
    minY = 0
    maxX = width - 1
    maxY = height - 1

    if x <= minX or y <= minY or w + x >= maxX or h + y >= maxY:
        return True
    return False


def FilterBorderContours(contours, imageSize):
    filtered_contoures = []
    for contour in contours:
        if not IsBorderContour(contour, imageSize):
            filtered_contoures.append(contour)
    return filtered_contoures


def Fill(input_image):
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


# READ ALL IMAGES
png_images = GetAllImages()


# CREATE DIRECTORY FOR RESULTS
CreateResultDirectory(png_images)


for file_name in png_images:
    # OPEN AND NORMALIZE
    image = cv2.imread(file_name, 0)
    normalized_image = (cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)) * 255
    cv2.imwrite(GetFileName("1norm", file_name), normalized_image)

    normalized_image = cv2.imread(GetFileName("1norm", file_name), 0)

    # FILTERING
    RADIUS = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RADIUS, RADIUS))
    opening = cv2.morphologyEx(normalized_image, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(GetFileName("2opening", file_name), opening)

    # SEGMENTATION

    # TODO set properly those values
    lower_thresh = 40
    upper_thresh = 3 * lower_thresh

    edges = cv2.Canny(opening, lower_thresh, upper_thresh)

    edges = cv2.blur(edges, (2, 2))   # TODO blur??? should it be here?
    cv2.imwrite(GetFileName("3canny_edges", file_name), edges)

    # FILL
    filled_image = Fill(edges)
    cv2.imwrite(GetFileName("4filled", file_name), filled_image)

    RADIUS_2 = 10
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RADIUS_2, RADIUS_2))
    opening_2 = cv2.morphologyEx(filled_image, cv2.MORPH_OPEN, kernel_2)
    cv2.imwrite(GetFileName("5opening", file_name), opening_2)

    # EXTRACTION
    contours, bin = cv2.findContours(opening_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    # CREATE PICTURE FOR DISPLAYING
    display_image = cv2.cvtColor(normalized_image.copy(), cv2.COLOR_GRAY2BGR)

    # DRAWING CONTOURS
    LINE_THICKNESS = 2
    cv2.drawContours(display_image, FilterBorderContours(contours, filled_image.shape), -1, YELLOW, LINE_THICKNESS)

    cv2.imshow("Result", display_image)

    cv2.imwrite(GetFileName("6result", file_name), display_image)
    cv2.waitKey(0)


cv2.destroyAllWindows()
