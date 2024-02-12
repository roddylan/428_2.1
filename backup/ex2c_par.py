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
        self.constants = {}
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
        plt.title("select 4 corners (static obj) (p1-p2 = bot. L - top L, p3-p4 = bot. R - top R)")
        points = plt.ginput(4, timeout=0)
        plt.close()
        hp1, hp2, hp3, hp4 = [np.array([pt[0], pt[1], 1]) for pt in points]
        l3 = np.cross(hp1.T, hp2.T).T
        l4 = np.cross(hp3.T, hp4.T).T
        self.constants["l3"] = l3
        self.constants["l4"] = l4
        self.const_pts = [hp1, hp2, hp3, hp4]


        plt.imshow(frame0[:,:,::-1])
        plt.title("select 4 corners (moving obj) (p1-p2 = bot. L - top L, p3-p4 = bot. R - top R)")
        points = plt.ginput(4, timeout=0)
        plt.close()

        self.tracks = [[],[],[],[]]
        
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
        t1, t2, t3, t4 = self.tracks
        dim = [self.Im[0].shape[1], self.Im[0].shape[0]]
        xs = np.arange(0, dim[0], dtype=np.float64)
        # ys = np.arange(0, dim[1], dtype=np.float64)
        
        self.create_folder(out)
        for i in range(len(self.cIm)):
            img = self.cIm[i]
            
            hp1 = np.array([t1[i][0] + t1[i][2]//2, t1[i][1] + t1[i][3]//2, 1]).reshape(3,1)
            hp2 = np.array([t2[i][0] + t2[i][2]//2, t2[i][1] + t2[i][3]//2, 1]).reshape(3,1)
            hp3 = np.array([t3[i][0] + t3[i][2]//2, t3[i][1] + t3[i][3]//2, 1]).reshape(3,1)
            hp4 = np.array([t4[i][0] + t4[i][2]//2, t4[i][1] + t4[i][3]//2, 1]).reshape(3,1)
            
            chp1 = self.const_pts[0]
            chp2 = self.const_pts[1]
            chp3 = self.const_pts[2]
            chp4 = self.const_pts[3]

            l1 = np.cross(hp1.T, hp2.T).reshape(3,)
            l2 = np.cross(hp3.T, hp4.T).reshape(3,)
            l3 = self.constants["l3"].reshape(3,)
            l4 = self.constants["l4"].reshape(3,)

            
            n_l1 = np.array([l1[0]/l1[2], l1[1]/l1[2], 1]).reshape(3,1)
            n_l2 = np.array([l2[0]/l2[2], l2[1]/l2[2], 1]).reshape(3,1)
            n_l3 = np.array([l3[0]/l3[2], l3[1]/l3[2], 1]).reshape(3,1)
            n_l4 = np.array([l4[0]/l4[2], l4[1]/l4[2], 1]).reshape(3,1)


            # err = np.cross(np.cross(l1.T, l2.T), np.cross(l3.T, l4.T))
            err = np.cross(np.cross(n_l1.T, n_l2.T), np.cross(n_l3.T, n_l4.T))
            e1,e2,e3 = err[0]
            cv2.putText(img, f"{np.linalg.norm(err):.10f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"{e1:.3e}, {e2:.3e}, {e3:.3e}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            # l1_ys = np.array([], dtype=np.float64)
            # l2_ys = np.array([], dtype=np.float64)
            # l3_ys = np.array([], dtype=np.float64)
            # l4_ys = np.array([], dtype=np.float64)
            # for x in xs:
            #     l1_y = -(l1[2] + l1[0] * x)/l1[1]
            #     l2_y = -(l2[2] + l2[0] * x)/l2[1]

            #     l3_y = -(l3[2] + l3[0] * x)/l3[1]
            #     l4_y = -(l4[2] + l4[0] * x)/l4[1]
                
            #     l1_ys = np.append(l1_ys, l1_y)
            #     l2_ys = np.append(l2_ys, l2_y)

            #     l3_ys = np.append(l3_ys, l3_y)
            #     l4_ys = np.append(l4_ys, l4_y)

            cv2.line(img, (int(hp1[0]), int(hp1[1])), (int(hp2[0]), int(hp2[1])), color=(0,0,255), thickness=2) # moving left
            cv2.line(img, (int(hp3[0]), int(hp3[1])), (int(hp4[0]), int(hp4[1])), color=(0,0,255), thickness=2) # moving right
            
            cv2.line(img, (int(chp1[0]), int(chp1[1])), (int(chp2[0]), int(chp2[1])), color=(0,0,255), thickness=2) # static left
            cv2.line(img, (int(chp3[0]), int(chp3[1])), (int(chp4[0]), int(chp4[1])), color=(0,0,255), thickness=2) # static right

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
    trackers = Trackers("2c_par_in")
    trackers.load_frames()
    
    trackers.setup()
    t = trackers.run_trackers()
    trackers.plot_trackers("2c_par_out")
    # trackers.plot_rect_trackers()



