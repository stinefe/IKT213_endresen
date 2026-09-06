import cv2
import numpy as np

def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    cv2.imwrite("solutions/iris-1_padded.jpg", padded_image)
    return padded_image

def crop(image, x0, x1, y0, y1):
    cropped_image = image[y0:y1, x0:x1]
    cv2.imwrite("solutions/iris-1_cropped.jpg", cropped_image)
    return cropped_image

def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite("solutions/iris-1_resized.jpg", resized_image)

def copy(image, emptyPictureArray):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            emptyPictureArray[y, x] = image[y, x]
    cv2.imwrite("solutions/iris-1_copy.jpg", emptyPictureArray)
    return emptyPictureArray

def grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("solutions/iris-1_grayscale.jpg", grayscale_image)
    return grayscale_image

def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("solutions/iris-1_hsv.jpg", hsv_image)
    return hsv_image

def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            emptyPictureArray[y, x] = image[y, x] + hue
    cv2.imwrite("solutions/iris-1_hueshifted.jpg", emptyPictureArray)
    return emptyPictureArray

def smoothing(image):
    smoothed_image = cv2.GaussianBlur(image, (15, 15), 0)
    cv2.imwrite("solutions/iris-1_smoothing.jpg", smoothed_image)
    return smoothed_image

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation_angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imwrite("solutions/iris-1_180_degrees.jpg", rotated_image)
    return rotated_image

def main():
    image = cv2.imread("iris-1.png")
    height, width, channels = image.shape
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)

    padding(image, 100)
    crop(image, 200,-130,200,-130)
    resize(image,200,200)
    copy(image, emptyPictureArray)
    grayscale(image)
    hsv(image)
    hue_shifted(image, emptyPictureArray, 50)
    smoothing(image)
    rotation(image, 90)
    rotation(image, 180)

if __name__ == '__main__':
    main()

