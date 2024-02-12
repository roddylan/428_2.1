import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import cv2, os
import shutil # for folders


'''
Given two images of a planar scene and a set of four or more 2D point correspondences between the images ( xi <--> x'i ), 
compute the homography, H, which relates one image to the other, and then apply H to the first image 
to see if you get similar image as the second one. 
'''




class DLT:
    def __init__(self, img1, img2, n_pts=4, l1=False, norm=False, eta0=1, max_it=50, img1_pts=None, img2_pts=None, thresh=np.float32(1e-10)):
        self.img1 = img1
        self.img2 = img2
        

        # self.img1_pts = []
        # self.img2_pts = []

        self.n_pts = n_pts
        self.l1 = l1 # use l1 
        self.norm = norm # 

        self.eta0 = eta0
        self.max_it = max_it

        self.H = None
        self.img1_pts = img1_pts
        self.img2_pts = img2_pts
        self.thresh = thresh

    def display(self, im1, im2):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.imshow(im1, cmap='gray')
        ax2.imshow(im2, cmap='gray')
        plt.show()

    def select(self):
        if self.img1_pts is None or self.img2_pts is None:
            # display 2 imgs
            fig, (ax1, ax2) = plt.subplots(1, 2)

            ax1.imshow(self.img1, cmap='gray')
            ax2.imshow(self.img2, cmap='gray')
            plt.title(f"select {self.n_pts} points in img1 (left)", loc='left')
            
            # plt.show()
            
            self.img1_pts = plt.ginput(self.n_pts, timeout=0)
            # print(type(self.img1_pts))
            plt.close()

            fig, (ax1, ax2) = plt.subplots(1, 2)

            ax1.imshow(self.img1, cmap='gray')
            ax2.imshow(self.img2, cmap='gray')
            plt.title(f"select {self.n_pts} corresponding points in img2 (right) (same order)")
            # plt.show()
            
            self.img2_pts = plt.ginput(self.n_pts)
            # print(self.img2_pts)
            plt.close()

    def get_T(self, pts):
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        w = np.max(xs) - np.min(xs)
        h = np.max(ys) - np.min(ys)

        T = np.linalg.inv(np.array([
            [w+h, 0, w/2],
            [0, w+h, h/2],
            [0, 0, 1]
        ]))
        pts = []
        for i in range(len(xs)):
            hpt = np.array([xs[i], ys[i], 1]).reshape(3,1)
            thpt = (T @ hpt).reshape(-1,)
            
            pts.append((thpt[0]/thpt[2], thpt[1]/thpt[2]))


        
        return T, pts
    
    def normalize(self):
        self.T1, self.nimg1_pts = self.get_T(self.img1_pts)
        self.T2, self.nimg2_pts = self.get_T(self.img2_pts)
        
    def denormalize(self):
        self.H = np.linalg.inv(self.T2) @ self.H @ self.T1
        
    def get_homog(self):
        # x' = Hx
        A = np.empty((0,9), np.float32)

        if not self.norm:
            img1_pts = self.img1_pts
            img2_pts = self.img2_pts
        else:
            img1_pts = self.nimg1_pts
            img2_pts = self.nimg2_pts
        
        for i in range(self.n_pts):
            # xp, yp, wp = self.img2_pts[i][0], self.img2_pts[i][1], 1
            xp, yp, wp = img2_pts[i][0], img2_pts[i][1], 1
            # XP = np.array([self.img2_pts[0], self.img2_pts[1], 1]).T # xp, yp, wp

            # x = np.array([self.img1_pts[i][0], self.img1_pts[i][1], 1]).reshape(3,1)
            x = np.array([img1_pts[i][0], img1_pts[i][1], 1]).reshape(3,1)

            z = np.zeros(3).reshape(1,3)
            
            yx = yp * x.T
            wx = wp * x.T
            xx = xp * x.T

            Ai_top = np.hstack((z, -wx, yx))
            Ai_bot = np.hstack((wx, z, -xx))

            Ai = np.vstack((Ai_top, Ai_bot))

            A = np.vstack((A, Ai))

        # _, _, V = np.linalg.svd(A)
        # h = V.T[:, -1]
        h = self.solve_svd(A)

        self.H = h.reshape(3, 3)

        if self.norm:
            self.denormalize()
        
        return self.H
    
    def solve_svd(self, A):
        _, _, V = np.linalg.svd(A)
        h = V.T[:, -1]
        return h

    def get_l1_homog(self):
        # print('L1')
        # img1_pts = self.img1_pts
        # img2_pts = self.img1_pts
        if not self.norm:
            img1_pts = self.img1_pts
            img2_pts = self.img2_pts
        else:
            img1_pts = self.nimg1_pts
            img2_pts = self.nimg2_pts

        A = np.empty((0,9), np.float32)
        for i in range(self.n_pts):
            # xp, yp, wp = self.img2_pts[i][0], self.img2_pts[i][1], 1
            xp, yp, wp = img2_pts[i][0], img2_pts[i][1], 1
            # XP = np.array([self.img2_pts[0], self.img2_pts[1], 1]).T # xp, yp, wp

            # x = np.array([self.img1_pts[i][0], self.img1_pts[i][1], 1]).reshape(3,1)
            x = np.array([img1_pts[i][0], img1_pts[i][1], 1]).reshape(3,1)

            z = np.zeros(3).reshape(1,3)
            
            yx = yp * x.T
            wx = wp * x.T
            xx = xp * x.T

            Ai_top = np.hstack((z, -wx, yx))
            Ai_bot = np.hstack((wx, z, -xx))

            Ai = np.vstack((Ai_top, Ai_bot))

            A = np.vstack((A, Ai))
        
        k = A.shape[0]
        eta = np.array([self.eta0]*k, dtype=np.float64).reshape(-1,)

        # iterate 
        h = np.ones(9)
        h = 1/np.linalg.norm(h) * h
        h = h.reshape(-1,1)


        dh = np.inf
        it = 0
        while dh > 0.1: 
            # update h
            eta[eta < self.thresh] = self.thresh
            # eta_i = np.array([1/n for n in eta], dtype=np.float64).reshape(-1,)
            rt_eta = np.array([1/np.sqrt(n) for n in eta], dtype=np.float64).reshape(-1,)

            D = np.diag(rt_eta)
            A_p = D @ A

            new_h = self.solve_svd(A_p).reshape(-1,1)

            dh = np.linalg.norm(new_h) / np.linalg.norm(h)
            h = new_h

            # update eta
            eta = np.abs(A @ h)
            # print(np.min(eta))
            # print(h.reshape(1,-1))
            
            # if (np.min(eta) < 0):
            #     break
            it += 1
            if it > self.max_it:
                break

        self.H = h.reshape(3,3)
        
        if self.norm:
            self.denormalize()
        
        return self.H

        # x' = Hx

    
    
    def transf_hom(self):
        
        self.warped = cv2.warpPerspective(self.img1, self.H, (self.img2.shape[1], self.img2.shape[0]))
        
        return self.warped


    def run(self):
        self.select()
        if self.norm:
            self.normalize()
        # self.get_homog()
        
        if not self.l1:
            self.get_homog()
        else:
            self.get_l1_homog()
        
        
        warped = self.transf_hom()
        
        self.display(warped, self.img2)

    def get_warped(self):
        return self.warped

    def get_pts(self):
        return self.img1_pts, self.img2_pts




# def compare_img(w1, w2, im):
    

if __name__ == "__main__":
    img1 = cv2.imread("3_in/key3.jpg")
    img2 = cv2.imread("3_in/key1.jpg")

    
    N_PTS = 6
    # 2a)
    dlt_a = DLT(img1, img2, n_pts=N_PTS)
    dlt_a.run()

    im1_pt, im2_pt = dlt_a.get_pts()

    # 2b)
    # dlt_b = DLT(img1, img2, n_pts=N_PTS, l1=True)
    dlt_b = DLT(img1, img2, n_pts=N_PTS, l1=True, img1_pts=im1_pt, img2_pts=im2_pt)
    dlt_b.run()

    # # 2c
    # dlt_c1 = DLT(img1, img2, n_pts=N_PTS, norm=True)
    dlt_c1 = DLT(img1, img2, n_pts=N_PTS, norm=True, img1_pts=im1_pt, img2_pts=im2_pt)
    dlt_c1.run()
    # im1_pt, im2_pt = dlt_c1.get_pts()

    dlt_c2 = DLT(img1, img2, n_pts=N_PTS, l1=True, norm=True, img1_pts=im1_pt, img2_pts=im2_pt)
    dlt_c2.run()



