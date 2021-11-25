import numpy as np
import matplotlib.pyplot as plt
from ctypes import *
import time


ll = cdll.LoadLibrary
lib = ll("gulugulu.dll")
lib.is_in_tri.restype = c_int

load_path='./transform/im1/'

def in_tria_c(p1x, p1y, p2x, p2y, p3x, p3y, px, py):
    px = (c_double)(px)
    py = (c_double)(py)
    return lib.is_in_tri(p1x, p1y, p2x, p2y, p3x, p3y, px, py)

def out_rect(p1x, p2x, p3x, p1y, p2y, p3y):

    xmax = max(p1x, max(p2x, p3x))
    xmin = min(p1x,min( p2x, p3x))
    ymax = max(p1y, max(p2y, p3y))
    ymin = min(p1y,min( p2y, p3y))
    ans = [(int)(xmin), (int)(xmax), (int)(ymin), (int)(ymax)]
    return ans

if __name__=="__main__":
    img_plt = plt.imread(load_path+'or.png')
    con = np.load(load_path+'s_contact.npy')
    pps = np.load(load_path+'s_poi.npy')
    tris=np.load(load_path+'s_tris.npy')
    print(tris.shape)
    tri_color=np.load(load_path+'s_tri_color.npy')
    print(tri_color.shape)
    pic_size =np.load(load_path+'s_pic_size.npy')
    print(pic_size)

    pic = np.ones(pic_size,dtype=np.float32)
    print(pic.shape)
    pic_ok = np.zeros(shape=[pic_size[0],pic_size[1]])
    plt.figure(0)
    tt=0
    for i,tri in enumerate(tris):
        i1, i2, i3 = tri[0], tri[1], tri[2]
        p1x, p2x, p3x, p1y, p2y, p3y = pps[i1, 0], pps[i2, 0], pps[i3, 0], pps[i1, 1], pps[i2, 1], pps[i3, 1]
        # plt.plot([p1x, p2x, p3x,p1x],[p1y, p2y, p3y,p1y])
        orect = out_rect(p1x, p2x, p3x, p1y, p2y, p3y)
        ymin, ymax, xmin, xmax = orect[0], orect[1], orect[2], orect[3]
        p1x, p2x, p3x, p1y, p2y, p3y = (c_double)(p1x), (c_double)(p2x), (c_double)(p3x), (c_double)(p1y), (c_double)(
            p2y), (c_double)(p3y)
        xmin, ymin, xmax, ymax = max(xmin, 0), max(ymin, 0), min(xmax, pic_size[0] - 1), min(ymax,pic_size[1] - 1)
        xmin, ymin, xmax, ymax = min(xmin, pic_size[0] - 1), min(ymin, ymax,pic_size[1] - 1), max(xmax, 0), max(ymax,0)
        # plt.plot([xmin,xmax,xmax,xmin,xmin],[ymin,ymin,ymax,ymax,ymin])
        # print(xmin, ymin, xmax, ymax )
        s1=time.time()
        for x in range(xmin, xmax):
            for y in range(ymin , ymax):
                if pic_ok[x,y]==1:
                    continue
                # if in_tria(pps[i1], pps[i2], pps[i3],y+0.5,x+0.5):
                if in_tria_c(p1x, p1y, p2x, p2y, p3x, p3y, y + 0.5, x + 0.5):
                    pic[x][y]=tri_color[i]
                    pic_ok[x][y] = 1
        s2=time.time()
        tt+=(s2-s1)
    # for p in pps:
    #     plt.plot(p[0],p[1],'*')
    # im_path='./transform/im1/or.png'
    # or_im=plt.imread(im_path)
    # plt.imshow(or_im)
    plt.ylim((0,pic_size[0]))
    plt.xlim((0,pic_size[1]))
    plt.imshow(img_plt)
    #
    # for p in pps:
    #     plt.plot(p[0],p[1],'r*')
    for i in range(len(pps)):
        for j in range(len(pps)):
            if con[i,j]==1:
                plt.plot([pps[i][0],pps[j][0]],[pps[i][1],pps[j][1]])


    plt.axis('off')
    plt.savefig(load_path+'im4.png' , dpi=500, bbox_inches='tight', pad_inches=0)
    print(tt)


    plt.show()


