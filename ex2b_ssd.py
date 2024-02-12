import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import cv2, os, time
import shutil # for folders


class Trackers:
    def __init__(self, src, max_it = 50, sz = 50):
        self.src = src
        self.MAX_ITERATION = max_it
        self.sz = sz
        self.Im = []
        self.trackers = []

    def load_frames(self):
        self.Im = []
        files = (os.listdir(self.src))
        # n = len(files)
        # for i in range(1,n+1):
        #     self.Im.append(cv2.imread(f"{self.src}/frame00001{i}.{type}", cv2.IMREAD_GRAYSCALE))

        for f in sorted(files):
            self.Im.append(cv2.imread(f"{self.src}/{f}", cv2.IMREAD_GRAYSCALE))

        return self.Im
    
    def select_point(self, frame0):
        plt.imshow(frame0, cmap="gray")
        plt.title("select 4 corners (p1-p2 = bot. L - top R, p3-p4 = bot. R - top L)")
        points = plt.ginput(4)
        plt.close()

        for point in points:
            self.trackers.append(Tracker(self.src, point, self.sz, self.Im))

    def load_video(self):
        self.Im = []
        
        cap = cv2.VideoCapture(self.src)
        
        while True:
            ret, img = cap.read()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if not ret:
                print("failed")
                break
            
            self.Im.append(img_gray)
        
        cv2.destroyAllWindows()
        cap.release()
        

    def setup(self):
        self.select_point(self.Im[0])
    
    def run_trackers(self):
        self.tracks = []
        for tracker in self.trackers:
            self.tracks.append(tracker.run())
        return self.tracks
    

    def plot_rect_trackers(self):
        t1, t2, t3, t4 = self.tracks
        dim = [self.Im[0].shape[1], self.Im[0].shape[0]]
        
        # fig = plt.figure()
        # fig.set_size_inches(1. * dim[0] / dim[1], 1, forward = False)
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        # fig, ax = plt.subplots()
        for i in range(len(self.Im)):
            img = self.Im[i]
            
            cv2.rectangle(img, (int(t1[i][0]), int(t1[i][1])), ( int(t1[i][0]+self.sz),  int(t1[i][1]+self.sz)), (0,0,255), 2)
            cv2.rectangle(img, (int(t2[i][0]), int(t2[i][1])), ( int(t2[i][0]+self.sz),  int(t2[i][1]+self.sz)), (0,0,255), 2)
            cv2.rectangle(img, (int(t3[i][0]), int(t3[i][1])), ( int(t3[i][0]+self.sz),  int(t3[i][1]+self.sz)), (0,0,255), 2)
            cv2.rectangle(img, (int(t4[i][0]), int(t4[i][1])), ( int(t4[i][0]+self.sz),  int(t4[i][1]+self.sz)), (0,0,255), 2)

            # p1 = t1[i]
            # p2 = t2[i]
            # p3 = t3[i]
            # p4 = t4[i]
            
            cv2.imshow("video", img)
            
            k = cv2.waitKey(1) # 1ms wait
            if k == 27:
                break

        
        cv2.destroyAllWindows()
        # cap.release()

    def plot_trackers(self, out=""):
        t1, t2, t3, t4 = self.tracks
        dim = [self.Im[0].shape[1], self.Im[0].shape[0]]
        xs = np.arange(0, dim[0], dtype=np.float64)
        # ys = np.arange(0, dim[1], dtype=np.float64)
        
        self.create_folder(out)
        for i in range(len(self.Im)):
            img = self.Im[i]

            hp1 = np.array([t1[i][0] + self.sz//2, t1[i][1] + self.sz//2, 1]).reshape(3,1)
            hp2 = np.array([t2[i][0] + self.sz//2, t2[i][1] + self.sz//2, 1]).reshape(3,1)
            hp3 = np.array([t3[i][0] + self.sz//2, t3[i][1] + self.sz//2, 1]).reshape(3,1)
            hp4 = np.array([t4[i][0] + self.sz//2, t4[i][1] + self.sz//2, 1]).reshape(3,1)
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
        

class Tracker:
    def __init__(self, src, pt, sz, im=[], max_it = 50, lvls = 2, sf = 1):
        self.src = src
        self.MAX_ITERATION = max_it
        
        self.Im = im
        
        self.lvls = lvls
        self.sf = sf
        
        self.xt = pt[0] - sz//2
        self.yt = pt[1] - sz//2
        self.w = sz
        self.h = sz
        
        self.p = np.array([0., 0.], dtype=np.float32)

        self.tracks = [[self.xt, self.yt]]

        self.dim = [self.Im[0].shape[1], self.Im[0].shape[0]]
        X = np.arange(0, self.dim[0], dtype=np.float32)
        Y = np.arange(0, self.dim[1], dtype=np.float32)
        self.Xq, self.Yq = np.meshgrid(X, Y)


    def get_pyramid(self, frame):
        pyr = [frame] # bottom -> top
        for i in range(self.lvls):
            next_lvl = pyr[i]
            for _ in range(self.sf):
                next_lvl = cv2.pyrDown(next_lvl)

            pyr.append(next_lvl)
        
        return pyr[::-1] # top -> bottom
    
    def pyr_ssd(self, frame, template, p, x0, y0, width, height):
        x = int(x0 + p[0])
        y = int(y0 + p[1])

        U = np.array([0., 0.], dtype=np.float32)
        u, v = np.inf, np.inf

        bbox = frame[y:y+height, x:x+width]
        
        max_it = self.MAX_ITERATION
        i=0
        
        while np.linalg.norm([u, v]) > 0.1 and i < max_it:
            i+=1
            r = np.float32(bbox) - np.float32(template)
            
            r[abs(r)<=20] = 0
            
            r_v = np.gradient(np.float64(bbox), axis = 0).flatten()
            r_u = np.gradient(np.float64(bbox), axis = 1).flatten()
            
            J_r = np.column_stack((r_u, r_v))
            
            s,_,_,_ = np.linalg.lstsq(J_r, -r.flatten())
            
            u = s[0]
            v = s[1]

            if (x0 + p[0] + u >= self.dim[0]) or (x0 + p[0] + u < 0):
                break
            if (y0 + p[1] + v >= self.dim[1]) or (y0 + p[1] + v < 0):
                break
            
            p[0] += u
            p[1] += v
            U[0] += u
            U[1] += v

            x = int(x0 + p[0])
            y = int(y0 + p[1])

            bbox = frame[y:y+height, x:x+width]

        
        return U
    
    def run(self):
        template = self.Im[0][int(self.yt):int(self.yt+self.h), int(self.xt):int(self.xt+self.w)]
        
        temp_pyr = self.get_pyramid(template)

        for frame in self.Im[1:]:
            frame_mapped = cv2.remap(frame, self.Xq, self.Yq, cv2.INTER_LINEAR)

            frame_pyr = self.get_pyramid(frame_mapped)
            U = np.array([0., 0.], dtype=np.float32)

            for i in range(self.lvls):
                curr_template = temp_pyr[i]
                curr_frame = frame_pyr[i]
                # if i < N_LVLS:
                P = (self.p // ((2**self.sf) ** (self.lvls - i))) + U
                x0 = self.xt // ((2**self.sf) ** (self.lvls - i))
                y0 = self.yt // ((2**self.sf) ** (self.lvls - i))
                # width = w // ((2**SCALE_FACTOR) * (N_LVLS - i))
                # height = h // ((2**SCALE_FACTOR) * (N_LVLS - i))
                # print(template.shape, w, h)
                width = curr_template.shape[1]
                height = curr_template.shape[0]
                
                motion = self.pyr_ssd(curr_frame, curr_template, P, x0, y0, width, height)
                U += motion
                
                
                # else:
                #     x0 = xt
                #     y0 = yt
                
                # U = ssd(curr_frame, curr_template, U)
                U = U * (2**self.sf)

            P = self.p + U
            # regular ssd
            U += self.pyr_ssd(frame_pyr[-1], temp_pyr[-1], P, self.xt, self.yt, self.w, self.h)
            self.p = self.p + U
            self.tracks.append([int(self.xt + self.p[0]), int(self.yt + self.p[1])])

        return self.tracks
    



if __name__ == "__main__":
    trackers = Trackers("2b_in")
    trackers.load_frames()
    trackers.setup()
    t = trackers.run_trackers()
    trackers.plot_trackers("2b_out")
    # trackers.plot_rect_trackers()


