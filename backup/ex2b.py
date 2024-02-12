import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import cv2, os, time
import shutil # for folders


class Trackers:
    def __init__(self, src, sz = 50):
        self.src = src
        self.sz = sz
        self.Im = []
        self.cIm = []
        self.trackers = []

    def load_frames(self):
        self.Im = []
        files = (os.listdir(self.src))

        for f in sorted(files):
            # self.cIm.append(cv2.imread(f"{self.src}/{f}", cv2.IMREAD_COLOR))
            # self.Im.append(cv2.imread(f"{self.src}/{f}", cv2.IMREAD_GRAYSCALE))

            img = cv2.imread(f"{self.src}/{f}", cv2.IMREAD_COLOR)
            g_img = cv2.imread(f"{self.src}/{f}", cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue

            self.cIm.append(img)
            self.Im.append(g_img)

        return self.Im
    
    def select_point(self, frame0):
        # plt.imshow(frame0, cmap="gray")
        plt.imshow(frame0[:,:,::-1])
        plt.title("select 4 corners (p1-p2 = bot. L - top R, p3-p4 = bot. R - top L)")
        points = plt.ginput(4, timeout=0)
        plt.close()

        self.tracks = [[],[],[],[]]
        
        for i, pt in enumerate(points):
            bbox = (np.int64(pt[0] - self.sz//2), np.int64(pt[1] - self.sz//2), np.int64(self.sz), np.int64(self.sz))
            self.tracks[i].append(bbox)
            
            # self.trackers.append(Tracker(self.src, pt, self.sz, self.Im))
            # tracker = cv2.TrackerKCF_create()
            # tracker.init(frame0, bbox)
            # tracker.init(frame0, np.int64(bbox))
            self.trackers.append(cv2.TrackerKCF_create())
            self.trackers[-1].init(frame0, bbox)

        self.dim = [frame0.shape[1], frame0.shape[0]]

    def load_video(self):
        self.Im = []
        
        cap = cv2.VideoCapture(self.src)
        
        while True:
            ret, img = cap.read()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if not ret:
                print("failed")
                break
            
            self.cIm.append(img)
            self.Im.append(img_gray)
        
        cv2.destroyAllWindows()
        cap.release()
        

    def setup(self):
        self.select_point(self.cIm[0])
    
    def run_trackers(self):
        X = np.arange(0, self.dim[0], dtype=np.float32)
        Y = np.arange(0, self.dim[1], dtype=np.float32)
        Xq, Yq = np.meshgrid(X, Y)

        for frame in self.cIm[1:]:
            
            for i in range(len(self.trackers)):
                frame_mapped = cv2.remap(frame, Xq, Yq, cv2.INTER_LINEAR)
                
                ret, bbox = self.trackers[i].update(frame_mapped)
                
                
                if ret:
                    self.tracks[i].append(bbox)
                else:
                    self.tracks[i].append(self.tracks[i][-1])

        return self.tracks

    

    def plot_trackers(self, out=""):
        t1, t2, t3, t4 = self.tracks
        dim = [self.Im[0].shape[1], self.Im[0].shape[0]]
        xs = np.arange(0, dim[0], dtype=np.float64)
        # ys = np.arange(0, dim[1], dtype=np.float64)
        
        self.create_folder(out)
        for i in range(len(self.Im)):
            img = self.Im[i]

            hp1 = np.array([t1[i][0] + t1[i][2]//2, t1[i][1] + t1[i][3]//2, 1]).reshape(3,1)
            hp2 = np.array([t2[i][0] + t2[i][2]//2, t2[i][1] + t2[i][3]//2, 1]).reshape(3,1)
            hp3 = np.array([t3[i][0] + t3[i][2]//2, t3[i][1] + t3[i][3]//2, 1]).reshape(3,1)
            hp4 = np.array([t4[i][0] + t4[i][2]//2, t4[i][1] + t4[i][3]//2, 1]).reshape(3,1)
            # hp2 = np.array([t2[i][0] + self.sz//2, t2[i][1] + self.sz//2, 1]).reshape(3,1)
            # hp3 = np.array([t3[i][0] + self.sz//2, t3[i][1] + self.sz//2, 1]).reshape(3,1)
            # hp4 = np.array([t4[i][0] + self.sz//2, t4[i][1] + self.sz//2, 1]).reshape(3,1)
            # cv2.circle(img, (int(hp1[0]), int(hp1[1])), 5, color=(0,0,255),thickness=5)
            # cv2.circle(img, (int(hp2[0]), int(hp2[1])), 5, color=(0,0,255),thickness=5)
            # cv2.circle(img, (int(hp3[0]), int(hp3[1])), 5, color=(0,0,255),thickness=5)
            # cv2.circle(img, (int(hp4[0]), int(hp4[1])), 5, color=(0,0,255),thickness=5)
            

            l1 = np.cross(hp1.T, hp2.T).T
            l2 = np.cross(hp3.T, hp4.T).T
            pm = np.cross(l1.T, l2.T).T

            # parallel line
            l3 = np.cross(hp1.T, hp4.T).T
            l4 = np.cross(hp2.T, hp3.T).T

            p_inf = np.cross(l3.T,l4.T).T

            lm = np.cross(p_inf.T, pm.T).T
            # l1_ys = np.array([], dtype=np.float64)
            # l2_ys = np.array([], dtype=np.float64)
            # l3_ys = np.array([], dtype=np.float64)
            # l4_ys = np.array([], dtype=np.float64)
            lm_ys = np.array([], dtype=np.float64)
            for x in xs:
                lm_y = -(lm[2] + lm[0] * x)/lm[1]
                lm_ys = np.append(lm_ys, lm_y)

            pmx, pmy = pm[0]/pm[2], pm[1]/pm[2]

            cv2.circle(img, (int(pmx), int(pmy)), 5, color=(0,0,255), thickness=-1)
            cv2.line(img, (int(xs[0]), int(lm_ys[0])), (int(xs[-1]), int(lm_ys[-1])), color=(0,0,255), thickness=2)

            cv2.imshow("video", img)
            self.output(out, img, i)
            
            k = cv2.waitKey(1) # 1ms wait
            if k == 27:
                break

        
        cv2.destroyAllWindows()

    def create_folder(self, out: str):
        if out == "":
            return
        try:
            shutil.rmtree(out)
            print("deleted")
        except:
            print("folder(s) dne")

        os.mkdir(out)
        print("folders created")
        
    def output(self, out: str, frame, i, name=""):
        if out == "":
            return

        cv2.imwrite(f"{out}/{name}{i:05d}.png", frame)
    



if __name__ == "__main__":
    trackers = Trackers("2b_in")
    trackers.load_frames()
    trackers.setup()
    t = trackers.run_trackers()
    trackers.plot_trackers("2b_out")
    # trackers.plot_rect_trackers()


