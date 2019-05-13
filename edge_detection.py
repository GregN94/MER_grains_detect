import math
import os
import cv2
import numpy as np

GREEN = (0, 200, 0)
BLUE = (200, 0, 0)
YELLOW = (51, 230, 230)
RED = (0, 0, 200)

DIR = "EdgeDetection"


def GetAllImages():
    return [dir for dir in os.listdir(".") if dir.endswith(".JPG")]


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


def GetBorderContours(contours, imageSize):
    border_contours = []
    for contour in contours:
        if IsBorderContour(contour, imageSize):
            border_contours.append(contour)
    return border_contours


def Fill(input_image):
    image_floodfill = input_image.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    height, width = input_image.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(image_floodfill, mask, (0, 150), 255)
    # Invert floodfilled image
    image_floodfill_inv = cv2.bitwise_not(image_floodfill)

    # Combine the two images to get the foreground.
    output_image = input_image | image_floodfill_inv
    return output_image


def DeleteContours(contours_to_delete, image):
    clean_image = image.copy()

    cv2.drawContours(clean_image, contours_to_delete, -1, (0,0,0), 3)
    cv2.drawContours(clean_image, contours_to_delete, -1, (0,0,0), cv2.FILLED)

    return clean_image


def DeleteSmallContours(image, size):

    clean_image = image.copy()
    contours, bin = cv2.findContours(clean_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    cont_to_delete = []
    for cont in contours:
        if cv2.contourArea(cont) < size:
            cont_to_delete.append(cont)

    return DeleteContours(cont_to_delete, clean_image)


def CalculateDistance(x, y):
    return math.sqrt(pow(x, 2) + pow(y, 2))


def CircleKernel(radius):

    diameter = 2 * radius + 1

    kernel = np.zeros((diameter, diameter), dtype=np.uint8)

    middle = int((diameter - 1) / 2)
    kernel[middle, ] = 1
    kernel[:, middle] = 1

    width, height = kernel.shape
    for x in range(width):
        for y in range(height):
            distance = round(CalculateDistance(abs(middle - x), abs(middle - y)))
            if distance <= radius:
                kernel[x, y] = 1
    return kernel


def DeleteBorderContours(image):
    clean_image = image.copy()
    contours, bin = cv2.findContours(clean_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    contours_to_delete = GetBorderContours(contours, clean_image.shape)

    return DeleteContours(contours_to_delete, clean_image)


# READ ALL IMAGES
png_images = GetAllImages()


# CREATE DIRECTORY FOR RESULTS
CreateResultDirectory(png_images)


for file_name in png_images:
    # OPEN AND NORMALIZE
    image = cv2.imread(file_name, 0)
    # normalized_image = (cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)) * 255
    # cv2.imwrite(GetFileName("1norm", file_name), normalized_image)

    # normalized_image = cv2.imread(GetFileName("1norm", file_name), 0)

    # FILTERING

    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, CircleKernel(8))
    cv2.imwrite(GetFileName("2opening", file_name), opening)

    # SEGMENTATION

    # TODO set properly those values
    ret3, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    RATIO = 0.5
    MULTIPLY = 1
    upper_thresh = MULTIPLY * ret3
    lower_thresh = RATIO * upper_thresh

    # WORKS BETTER WITH LITTLE FILTERING
    opening = cv2.bilateralFilter(opening, 10, 50, 50)

    edges = cv2.Canny(opening, lower_thresh, upper_thresh, 3, L2gradient=True)
    cv2.imshow("edges", edges)

    cv2.imwrite(GetFileName("3canny_edges", file_name), edges)

    # FILL
    filled_image = Fill(edges)

    image_with_no_border = DeleteBorderContours(filled_image)
    cv2.imwrite(GetFileName("4filled", file_name), image_with_no_border)
    cv2.imshow("filled", image_with_no_border)

    image_with_no_border = DeleteSmallContours(image_with_no_border, 300)
    cv2.imwrite(GetFileName("5deleted", file_name), image_with_no_border)

    RADIUS_2 = 10
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RADIUS_2, RADIUS_2))
    opening_2 = cv2.morphologyEx(image_with_no_border, cv2.MORPH_OPEN, kernel_2)
    cv2.imwrite(GetFileName("6opening", file_name), opening_2)

    # EXTRACTION
    contours, bin = cv2.findContours(opening_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    # CREATE PICTURE FOR DISPLAYING
    display_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

    # DRAWING CONTOURS
    LINE_THICKNESS = 2
    cv2.drawContours(display_image, contours, -1, YELLOW, LINE_THICKNESS)

    # cv2.imshow("Result", display_image)

    cv2.imwrite(GetFileName("7result", file_name), display_image)
    cv2.waitKey(0)


cv2.destroyAllWindows()
