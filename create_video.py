import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import cv2, os, time
import shutil # for folders

def create_folder():
    try:
        os.mkdir("vids")
    except:
        print('failed to create new folder')


def create_video(src, name, fps=24):
    images = [img for img in sorted(os.listdir(src))]
    frame = cv2.imread(os.path.join(src, images[0]))
    h, w, l = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid = cv2.VideoWriter(f"vids/{name}", fourcc, fps, (w, h))

    for img in images:
        vid.write(cv2.imread(os.path.join(src, img)))

    cv2.destroyAllWindows()
    vid.release()


if __name__ == "__main__":
    # create_folder()
    # create_video("2c_par_out", "2c_par_out.mp4")
    pass