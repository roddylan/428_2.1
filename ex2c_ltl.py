import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import cv2, os, time
import shutil # for folders


class Trackers:
    def __init__(self, src, sz = 30):
        self.src = src
        self.sz = sz
        self.Im = []
        self.cIm = []
        self.trackers = []
        self.const_pts = []

    def load_frames(self):
        self.Im = []
        files = (os.listdir(self.src))
        # n = len(files)
        # for i in range(1,n+1):
        #     self.Im.append(cv2.imread(f"{self.src}/frame00001{i}.{type}", cv2.IMREAD_GRAYSCALE))

        for f in sorted(files):
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
        
        plt.title("select 2 points (surface) L-R")
        points = plt.ginput(2, timeout=0)
        plt.close()
        hp1, hp2 = [np.array([pt[0], pt[1], 1]) for pt in points]
        self.l2 = np.cross(hp1.T, hp2.T).T
        
        self.const_pts = [hp1, hp2]


        plt.imshow(frame0[:,:,::-1])
        plt.title("select 2 points (obj) L-R")
        points = plt.ginput(2, timeout=0)
        plt.close()

        self.tracks = [[],[]]
        
        for i, pt in enumerate(points):
            bbox = (np.int64(pt[0] - self.sz//2), np.int64(pt[1] - self.sz//2), np.int64(self.sz), np.int64(self.sz))
            self.tracks[i].append(bbox)
            
            # self.trackers.append(cv2.TrackerKCF_create())
            self.trackers.append(cv2.TrackerCSRT_create())
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
        t1, t2 = self.tracks
        dim = [self.Im[0].shape[1], self.Im[0].shape[0]]
        xs = np.arange(0, dim[0], dtype=np.float64)
        # ys = np.arange(0, dim[1], dtype=np.float64)
        
        self.create_folder(out)
        for i in range(len(self.cIm)):
            img = self.cIm[i]
            
            hp1 = np.array([t1[i][0] + t1[i][2]//2, t1[i][1] + t1[i][3]//2, 1]).reshape(3,1)
            hp2 = np.array([t2[i][0] + t2[i][2]//2, t2[i][1] + t2[i][3]//2, 1]).reshape(3,1)
            
            
            hp3 = self.const_pts[0].reshape(3,1)
            hp4 = self.const_pts[1].reshape(3,1)

            hs = np.hstack((hp1, hp2, hp3, hp4))
            m = np.max(hs)

            nhp1 = hp1 / m
            nhp2 = hp2 / m
            nhp3 = hp3 / m
            nhp4 = hp4 / m
            

            # l1 = np.cross(hp1.T, hp2.T).reshape(3,)
            # l2 = self.l2.reshape(3,)


            

            # err = hp1 . (hp3 x hp4) + hp2 . (hp3 x hp4)
            # c = np.cross(hp3.T, hp4.T).T
            c = np.cross(nhp3.T, nhp4.T).T

            err = np.dot(nhp1.T, c) + np.dot(nhp2.T, c)
            
            
            cv2.putText(img, f"{np.linalg.norm(err):.10f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            

            cv2.line(img, (int(hp1[0]), int(hp1[1])), (int(hp2[0]), int(hp2[1])), color=(0,0,255), thickness=2) # obj
            cv2.line(img, (int(hp3[0]), int(hp3[1])), (int(hp4[0]), int(hp4[1])), color=(0,0,255), thickness=2) # surface
            
            

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
    trackers = Trackers("2c_ltl_in")
    trackers.load_frames()
    
    trackers.setup()
    t = trackers.run_trackers()
    trackers.plot_trackers("2c_ltl_out")
    # trackers.plot_rect_trackers()



