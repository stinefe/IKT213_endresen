import cv2

def print_image_information(image):
    """prints the information about the image"""
    height, width, channels = image.shape
    print("height: ", height)
    print("width: ", width)
    print("channels: ", channels)
    print("size: ", image.size)
    print("data type: ", image.dtype)

def save_camera_information():
    """saves the camera information into a .txt file"""
    camera = cv2.VideoCapture(0)
    fps = camera.get(cv2.CAP_PROP_FPS)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

    with open ("solutions/camera_outputs.txt", "w") as f:
        f.write(f"fps: {fps}\n")
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")

    camera.release()

def main():
    image = cv2.imread("iris-1.jpg")
    print_image_information(image)
    save_camera_information()

if __name__ == '__main__':
    main()