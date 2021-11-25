import matplotlib.pyplot as plt
import numpy as np
import deNet
from ctypes import *
import time

core = np.array([[-1.4142, -1, -1.4142], [-1, 9.6569, -1], [-1.4142, -1, -1.4142]])
ll = cdll.LoadLibrary
lib = ll("gulugulu.dll")
lib.is_in_tri.restype = c_int


# lib.cal_v_c.restype = c_double

def im2col(m, h, w, r=False):
    col, row = m.shape
    s1 = col - h + 1
    s2 = row - w + 1
    s = s1 * s2

    arr = np.array(m, dtype=np.float32)
    arr = arr.ctypes.data_as(POINTER(c_float))

    ans = (c_float * (s * h * w))(*range(s * h * w))

    lib.im2col(arr, col, row, h, w, ans)

    bns = np.ctypeslib.as_array(ans)
    if r:
        bns = bns.reshape([s, h * w])
    return bns


def filter(m, core):
    bns = 0
    h, w = 3, 3
    col, row, hig = m.shape
    c = core.reshape(9, 1)
    for i in range(m.shape[2]):
        arr = m[:, :, i]
        A = im2col(arr, h, w, r=True)
        ans = np.matmul(A, c)
        # print(ans.shape)
        bns += np.square(ans)
    bns = bns.reshape([col - 2, row - 2])
    return bns


def in_tria_c(p1x, p1y, p2x, p2y, p3x, p3y, px, py):
    px = (c_double)(px)
    py = (c_double)(py)
    return lib.is_in_tri(p1x, p1y, p2x, p2y, p3x, p3y, px, py)


def out_rect(p1x, p2x, p3x, p1y, p2y, p3y):
    # xmax = np.max(np.array([p1x, p2x, p3x]))
    # xmin = np.min(np.array([p1x, p2x, p3x]))
    # ymax = np.max(np.array([p1y, p2y, p3y]))
    # ymin = np.min(np.array([p1y, p2y, p3y]))
    # ans = np.array([xmin, xmax, ymin, ymax])
    # ans = np.floor(ans)
    # ans = np.array(ans, dtype=np.int64)
    # return ans

    xmax = max(p1x, max(p2x, p3x))
    xmin = min(p1x, min(p2x, p3x))
    ymax = max(p1y, max(p2y, p3y))
    ymin = min(p1y, min(p2y, p3y))
    ans = [(int)(xmin), (int)(xmax), (int)(ymin), (int)(ymax)]
    return ans


def cal_v(m1):  # m1(3*3*3)
    r = np.sum(m1[:, :, 0] * core)
    g = np.sum(m1[:, :, 1] * core)
    b = np.sum(m1[:, :, 2] * core)
    return r * r + g * g + b * b


def run(path_truth, output_path, black):
    start_time = time.time()
    path_truth = path_truth
    output_path = output_path
    img_plt = plt.imread(path_truth)
    print("图片大小:", img_plt.shape)
    # 去黑边
    if black:
        img_plt = img_plt[101:-100, :, :]
    black1 = np.zeros(shape=[101, img_plt.shape[1], 3])
    black2 = np.zeros(shape=[100, img_plt.shape[1], 3])
    if (img_plt.shape[2] == 4):
        img_plt = img_plt[:, :, :-1]
    plt.figure(0)
    # plt.imshow(img_plt)

    edge = np.zeros(shape=[img_plt.shape[0], img_plt.shape[1]])
    # print("正在计算边缘")
    # shape0 = img_plt.shape[0]
    # shape1 = img_plt.shape[1]
    # for i in range(1,shape0 - 1):
    #     for j in range(1,shape1 - 1):
    #         edge[i, j] = cal_v(img_plt[i - 1:i + 2, j - 1:j + 2, :])

    edge[1:-1, 1:-1] += filter(img_plt, core)

    pnum = (img_plt.shape[0] + img_plt.shape[1])//2
    # pnum = 100
    count = 0
    door = 0.05
    frame1 = time.time()
    print("loop0=", frame1 - start_time)
    for i in range(img_plt.shape[0]):
        for j in range(img_plt.shape[1]):
            if edge[i, j] > door:
                count += 1
    rate = np.random.rand(img_plt.shape[0], img_plt.shape[1])
    rate = (rate < pnum / count) - 0
    edge = edge * rate
    frame2 = time.time()
    print("loop1=", frame2 - frame1)

    # print("潜在顶点数=",count)
    c = 0
    for i in range(img_plt.shape[0]):
        for j in range(img_plt.shape[1]):
            if edge[i, j] > door:
                c += 1
                plt.plot(j, i, '*')
    pps = np.zeros(shape=[c + 3, 2])
    # print("实际顶点数约为", c)
    frame3 = time.time()
    print("loop2=", frame3 - frame2)

    k = 3
    for i in range(img_plt.shape[0]):
        for j in range(img_plt.shape[1]):
            if edge[i, j] > door:
                pps[k, 0] = j
                pps[k, 1] = i
                k += 1

    # 另外，需要补几个比较靠外的点使得整个图上都有三角形分布
    p4s = [[0, 0], [0, img_plt.shape[1] - 1], [img_plt.shape[0] - 1, 0], [img_plt.shape[0] - 1, img_plt.shape[1] - 1]]
    p4s = np.array(p4s)
    pps = np.concatenate([pps, p4s], axis=0)

    xmin = np.min(pps[3:, 0])
    xmax = np.max(pps[3:, 0])
    ymin = np.min(pps[3:, 1])
    ymax = np.max(pps[3:, 1])

    # 找到超级三角形,代替点集合的前三个点
    a = xmax - xmin
    b = ymax - ymin
    p1x = xmin - b
    p2x = xmax + b
    p3x = xmin + a / 2.0
    p1y = ymin
    p2y = ymin
    p3y = ymax + a / 2.0
    pps[0, 0] = p1x
    pps[0, 1] = p1y
    pps[1, 0] = p2x
    pps[1, 1] = p2y
    pps[2, 0] = p3x
    pps[2, 1] = p3y

    # print(pps)
    frame35 = time.time()
    print("loop3=", frame35 - frame3)

    # print("start calculating triangles...")
    [con, tris] = deNet.delaunay_net(pps, bigtri=True)
    s_con = np.array(con)
    frame4 = time.time()
    print("loop3=", frame4 - frame35)

    # print("connecting...")
    # for i in range(pps.shape[0]):
    #     for j in range(pps.shape[0]):
    #         if i > j and con[i, j] == 1:
    #             plt.plot([pps[i][0], pps[j][0]], [pps[i][1], pps[j][1]], 'k')

    # print("三角形个数=",len(tris))
    frame5 = time.time()
    print("loop4=", frame5 - frame4)

    pic = np.ones_like(img_plt)
    pic_ok = np.zeros(shape=[img_plt.shape[0], img_plt.shape[1]])
    # print("颜色填充中……")
    tt = 0
    tri_color = []
    for tri in tris:
        t1 = time.time()
        i1, i2, i3 = tri[0], tri[1], tri[2]
        p1x, p2x, p3x, p1y, p2y, p3y = pps[i1, 0], pps[i2, 0], pps[i3, 0], pps[i1, 1], pps[i2, 1], pps[i3, 1]
        orect = out_rect(p1x, p2x, p3x, p1y, p2y, p3y)
        ymin, ymax, xmin, xmax = orect[0], orect[1], orect[2], orect[3]
        p1x, p2x, p3x, p1y, p2y, p3y = (c_double)(p1x), (c_double)(p2x), (c_double)(p3x), (c_double)(p1y), (c_double)(
            p2y), (c_double)(p3y)
        # plt.plot([xmin,xmax,xmax,xmin,xmin],[ymin,ymin,ymax,ymax,ymin])
        xmin, ymin, xmax, ymax = max(xmin, 0), max(ymin, 0), min(xmax, img_plt.shape[0] - 1), min(ymax,
                                                                                                  img_plt.shape[1] - 1)
        pix_num = 0

        t2 = time.time()
        color = np.array([0, 0, 0], dtype=float)
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                if pic_ok[x, y] == 2:
                    continue
                # if in_tria(pps[i1], pps[i2], pps[i3],y+0.5,x+0.5):
                if in_tria_c(p1x, p1y, p2x, p2y, p3x, p3y, y + 0.5, x + 0.5):
                    pix_num += 1
                    color += img_plt[x, y]
                    pic_ok[x][y] = 1

        if pix_num == 0:
            tri_color.append(color)
            continue
        color /= pix_num
        tri_color.append(color)
        t3 = time.time()

        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                if pic_ok[x, y] == 1:
                    pic[x, y] = color
                    pic_ok[x, y] = 2

        t4 = time.time()
        tt += t3 - t2
    print(tt)
    frame6 = time.time()
    print("loop5=", frame6 - frame5)

    s_poi = pps
    s_tris = np.array(tris)
    s_tri_color = np.array(tri_color)
    s_pic_size = np.array(pic.shape)

    savepath = './transform/im1/'
    np.save(savepath + 's_poi', s_poi)
    np.save(savepath + 's_tris', s_tris)
    np.save(savepath + 's_tri_color', s_tri_color)
    np.save(savepath + 's_pic_size', s_pic_size)
    np.save(savepath + 's_contact', s_con)

    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if np.sum(pic[i, j, :]) == 3:
                pic[i, j] = img_plt[i, j]

    # img_plt=np.concatenate([black1,img_plt],axis=0)
    # img_plt=np.concatenate([img_plt,black2],axis=0)
    # plt.ylim((0,img_plt.shape[0]))
    # plt.xlim((0,img_plt.shape[1]))
    # plt.gca().invert_yaxis()
    # plt.axis('off')
    # plt.savefig(output_path+imgname+"_tri.png",dpi=500, bbox_inches='tight', pad_inches=0)
    plt.figure(1)
    if black:
        pic = np.concatenate([black1, pic], axis=0)
        pic = np.concatenate([pic, black2], axis=0)
    plt.imshow(pic)
    plt.axis('off')
    plt.savefig(output_path, dpi=500, bbox_inches='tight', pad_inches=0)
    # print("done!")
    end_time = time.time()
    print("填充完毕！ 耗时 time cost=", end_time - start_time)

    # plt.show()


if __name__ == "__main__":

    for i in range(6, 7):
        # imgname = 'f'+str(i)
        # print("now is ",imgname)
        # path_truth = './video/pics/' + imgname + '.png'
        # output_path = './res/output/' + imgname + "_done.png"
        imgname = 's' + str(i)
        print("now is ", i)
        path_truth = './image/show/' + imgname + '.png'
        output_path = './image/show/' + imgname + "_try2.png"
        run(path_truth, output_path, black=False)
