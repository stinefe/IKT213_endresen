import cv2
import numpy as np

def sobel_edge_detection(image):
    blurred_image = cv2.GaussianBlur(image,(3,3),0)
    sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=1)
    sobelxy = cv2.magnitude(sobelx, sobely)
    cv2.imwrite("solutions/sobel_edge_image.png", cv2.convertScaleAbs(sobelxy))

def canny_edge_detection(image, threshold_1, threshold_2):
    blurred_image = cv2.GaussianBlur(image,(3,3),0)
    canny = cv2.Canny(blurred_image, threshold_1, threshold_2)
    cv2.imwrite("solutions/canny_edge_image.png", canny)

def template_match(image, template):

    shapes_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w,h = template_gray.shape[::-1]

    res = cv2.matchTemplate(shapes_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv2.imwrite("solutions/template_match.png", image)

def resize(image, scale_factor: int, up_or_down: str):
    rows, cols, _channels = map(int, image.shape)

    if up_or_down == "up":
        resized_image = cv2.pyrUp(image, dstsize=(scale_factor * cols, scale_factor * rows))
        cv2.imwrite("solutions/resized_image_up.png", resized_image)
    elif up_or_down == "down":
        resized_image = cv2.pyrDown(image, dstsize=(cols // scale_factor, rows // scale_factor))
        cv2.imwrite("solutions/resized_image_down.png", resized_image)
    else:
        print("Invalid up_or_down input")
        return

def main():
    lambo = cv2.imread("lambo.png")
    shapes = cv2.imread("shapes-1.png")
    shapes_template = cv2.imread("shapes_template.jpg")

    sobel_edge_detection(lambo)
    canny_edge_detection(lambo, 50, 50)
    template_match(shapes, shapes_template)

    resize(lambo, 2, "up")
    resize(lambo, 2, "down")

if __name__ == "__main__":
    main()