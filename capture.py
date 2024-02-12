import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import cv2, os, time
import shutil # for folders

FPS_LIMIT = 24

# a) Capture Images
def clear(cam: cv2.VideoCapture):
    # release cam and clear windows
    cam.release()
    cv2.destroyAllWindows()

def capture_video(cam: cv2.VideoCapture, path):
    Im = []
    # cam = cv2.VideoCapture(1)
    cv2.namedWindow("capture")
    
    count = 0
    # IMG_PATH = "imgs"

    # folder clean up
    G_IMG_PATH = path
    try:
        shutil.rmtree(G_IMG_PATH)
        # shutil.rmtree('video_pyr')
        print("deleted")
    except:
        print("folder(s) dne")

    os.mkdir(G_IMG_PATH)
    # os.mkdir("video_pyr")
    print("folders created")
    
    

    prev = 0
    recording = False


    while True:
        ret, img = cam.read()
        
        if not ret:
            print("failed")
            break
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("capture", img)

        k = cv2.waitKey(1) # 1ms wait
        
        if k % 256 == 32: # space
            recording = True
        
        
        if (time.time() - prev > 1./FPS_LIMIT and recording):
            # Im.append(img_gray)
            count += 1
            cv2.imwrite(f"{G_IMG_PATH}/{count:05d}.png", img_gray)
            prev = time.time()

        if k % 256 == 8 or k % 256 == 27:
            # backspace or esc
            break
    
    # cam.release()
    clear(cam)
    
    return Im


if __name__ == "__main__":
    c = cv2.VideoCapture(1)

    capture_video(c, "2c_par_in")