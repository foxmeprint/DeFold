import matplotlib.pyplot as plt
import numpy as np
import time
from ctypes import *


def outheart(p1, p2, p3, ps):
    '''

    :param p1: 第一个点的索引
    :param p2: 第二个点的索引
    :param p3: 第三个点的索引
    :return: 外心的坐标
    '''
    A = np.array([[2 * (ps[p1, 0] - ps[p2, 0]), 2 * (ps[p1, 1] - ps[p2, 1])], \
                  [2 * (ps[p1, 0] - ps[p3, 0]), 2 * (ps[p1, 1] - ps[p3, 1])]])
    B = np.array([[ps[p1, 0] ** 2 - ps[p2, 0] ** 2 + ps[p1, 1] ** 2 - ps[p2, 1] ** 2], \
                  [ps[p1, 0] ** 2 - ps[p3, 0] ** 2 + ps[p1, 1] ** 2 - ps[p3, 1] ** 2]])
    Ai = np.linalg.inv(A)
    # print(Ai.shape,B.shape)
    temp = np.matmul(Ai, B)
    return temp.squeeze()


def is_in_cir(p1, p2, p3, p, ps):
    '''

    :param p1: 第1个点的索引
    :param p2: 第2个点的索引
    :param p3: 第3个点的索引
    :param p: 所求点的索引
    :return:在外接圆内则返回True,否则False
    '''
    # A = np.array([[2 * (ps[p1, 0] - ps[p2, 0]), 2 * (ps[p1, 1] - ps[p2, 1])], \
    #               [2 * (ps[p1, 0] - ps[p3, 0]), 2 * (ps[p1, 1] - ps[p3, 1])]])
    # if(np.linalg.det(A)==0):
    #     return False

    if ((ps[p1, 0] - ps[p2, 0]) * (ps[p1, 1] - ps[p3, 1]) - (ps[p1, 0] - ps[p3, 0]) * (ps[p1, 1] - ps[p2, 1]) == 0):
        return False
    out = outheart(p1, p2, p3, ps)
    dis0 = (out[0] - ps[p1, 0]) ** 2 + (out[1] - ps[p1, 1]) ** 2
    dis1 = (out[0] - ps[p2, 0]) ** 2 + (out[1] - ps[p2, 1]) ** 2
    dis3 = (out[0] - ps[p, 0]) ** 2 + (out[1] - ps[p, 1]) ** 2
    if dis3 > dis0 or dis3 > dis1:
        return False
    else:
        return True


def niclock(p1: np.array, p2: np.array, p3x: np.float64, p3y: np.float64) -> np.float64:
    # p1p2x = p[1,0] - p[0,0]
    # p1p2y = p[1,1] - p[0,1]
    # p1p3x = p[2,0] - p[0,0]
    # p1p3y = p[2,1] - p[0,1]
    return (p2[0] - p1[0]) * (p3y - p1[1]) - (p2[1] - p1[1]) * (p3x - p1[0])


def in_tria(p1: np.array, p2: np.array, p3: np.array, x: np.float64, y: np.float64) -> bool:
    if niclock(p1, p2, p3[0], p3[1]) < 0:
        return in_tria(p1, p3, p2, x, y)
    if niclock(p1, p2, x, y) > 0:
        if niclock(p2, p3, x, y) > 0:
            if niclock(p3, p1, x, y) > 0:
                return True
    return False


def chacheng(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def out_tri(p1, p2, p3, p, ps, contact, gro):
    v1 = ps[p] - ps[p1]
    v2 = ps[p] - ps[p2]
    v3 = ps[p] - ps[p3]
    a1 = p1
    a2 = p2
    a3 = p3
    if chacheng(v1, v2) * chacheng(v1, v3) < 0:
        a1, a2, a3 = p2, p3, p1

    if chacheng(v2, v3) * chacheng(v2, v1) < 0:
        a1, a2, a3 = p1, p3, p2
    contact[a1, a2] = 0
    contact[a2, a1] = 0
    contact[p1, p] = 1
    contact[p2, p] = 1
    contact[p3, p] = 1
    contact[p, p1] = 1
    contact[p, p2] = 1
    contact[p, p3] = 1
    gro.append([a2, a3, p])
    gro.append([a1, a3, p])


def delaunay_net(ps, bigtri):
    # 存放当前所有三角形的list
    tris = []
    # 将超级三角形的三个顶点存入list里
    temp = [0, 2, 1]
    temp.sort()
    tris.append(temp)
    # 每两个点之间的链接矩阵
    contact = np.zeros(shape=[ps.shape[0], ps.shape[0]])
    # 超级三角形的3个点相连
    for i in range(3):
        for j in range(3):
            if i != j: contact[i, j] = 1
    # print(contact)

    # 添加点进去
    for i in range(ps.shape[0]):
        # 跳过前三个点，因为他们是超级三角形
        if i < 3: continue
        t = []
        # 检查有没有不合法的三角形，主要是三点共线的三角
        for sanjiao in tris:
            a11 = ps[sanjiao[0], 0] - ps[sanjiao[1], 0]
            a12 = ps[sanjiao[0], 1] - ps[sanjiao[1], 1]
            a21 = ps[sanjiao[0], 0] - ps[sanjiao[2], 0]
            a22 = ps[sanjiao[0], 1] - ps[sanjiao[2], 1]
            if (a11 * a22) - (a12 * a21) == 0:
                t.append(sanjiao)
        # 删除所有不合法的三角
        for sanjiao in t:
            tris.remove(sanjiao)
        # 垃圾桶
        trash = []
        # 反垃圾桶
        gro = []
        for san in tris:
            if not is_in_cir(san[0], san[1], san[2], i, ps):
                # 无事发生
                continue
            # 三角在三角外
            if not in_tria(np.array([ps[san[0], 0], ps[san[0], 1]]), \
                           np.array([ps[san[1], 0], ps[san[1], 1]]), \
                           np.array([ps[san[2], 0], ps[san[2], 1]]), \
                           ps[i, 0], ps[i, 1]):
                # 重新分配这4个点的链接
                # 更新contac关系
                out_tri(san[0], san[1], san[2], i, ps, contact, gro)
                # 更新三角list（去一个加两个）
                trash.append(san)
                continue
            # 链接三角形每一个顶点与第4个点
            contact[san[0], i] = 1
            contact[san[1], i] = 1
            contact[san[2], i] = 1
            contact[i, san[0]] = 1
            contact[i, san[1]] = 1
            contact[i, san[2]] = 1
            # 更新三角list（去一个加三个）
            trash.append(san)
            gro.append([san[0], san[1], i])
            # print("!!")
            gro.append([san[2], san[1], i])
            # print("!!!")
            gro.append([san[0], san[2], i])
            # print("!!!!")
        for t1 in trash:
            tris.remove(t1)
        for g in gro:
            # print("!")
            tris.append(g)
        # print(trash)
        # print(gro)
        # print("tri=",tris)
        bin = []
        for t in tris:
            if contact[t[0], t[1]] == 0 or \
                    contact[t[0], t[2]] == 0 or \
                    contact[t[0], t[2]] == 0:
                bin.append(t)
        for b in bin:
            tris.remove(b)
    # print("tri2=", tris)

    if not bigtri:
        for i in range(ps.shape[0]):
            contact[i, 0] = 0
            contact[i, 1] = 0
            contact[i, 2] = 0
            contact[0, i] = 0
            contact[1, i] = 0
            contact[2, i] = 0

        trash = []
        for t in tris:
            if (0 in t) or (1 in t) or (2 in t):
                trash.append(t)
                continue
        for t in trash:
            tris.remove(t)
    # plt.figure(5)
    #
    # for i in range(ps.shape[0]):
    #     for j in range(ps.shape[0]):
    #         if i > j and contact[i, j] == 1:
    #             plt.plot([ps[i][0], ps[j][0]], [ps[i][1], ps[j][1]], 'k')
    #
    # plt.show()

    return [contact, tris]


if __name__ == "__main__":

    mapsizeX = 500
    mapsizeY = 500
    points_num = 100

    fr1 = time.time()
    pointsX = np.random.rand(points_num + 3, 1) * mapsizeX
    pointsY = np.random.rand(points_num + 3, 1) * mapsizeY

    # radius=50
    # outcir=16
    # theta=np.random.rand(points_num + 3+outcir, 1) * 2*np.pi
    # rho=np.random.rand(points_num + 3, 1) * radius
    #
    # pointsX = rho*np.cos(theta[:-outcir])
    # pointsY = rho*np.sin(theta[:-outcir])
    # cirX=np.ones(shape=[outcir,1])*np.cos(theta[-outcir:])*radius
    # cirY=np.ones(shape=[outcir,1])*np.sin(theta[-outcir:])*radius
    # pointsX=np.concatenate([pointsX,cirX],axis=0)
    # pointsY=np.concatenate([pointsY,cirY],axis=0)

    ps = np.concatenate([pointsX, pointsY], axis=1)
    # print(points)
    # 找到超级三角形,代替点集合的前三个点
    xmin = np.min(pointsX)
    xmax = np.max(pointsX)
    ymin = np.min(pointsY)
    ymax = np.max(pointsY)

    a = xmax - xmin
    b = ymax - ymin
    p1x = xmin - b
    p2x = xmax + b
    p3x = xmin + a / 2.0
    p1y = ymin
    p2y = ymin
    p3y = ymax + a / 2.0
    ps[0, 0] = p1x
    ps[0, 1] = p1y
    ps[1, 0] = p2x
    ps[1, 1] = p2y
    ps[2, 0] = p3x
    ps[2, 1] = p3y

    plt.figure(0)

    plt.axis('off')
    plt.plot(pointsX[3:], pointsY[3:], '*')

    print(ps.shape)

    [c, d] = delaunay_net(ps, bigtri=False)

    for i in range(ps.shape[0]):
        for j in range(ps.shape[0]):
            if i > j and c[i, j] == 1:
                plt.plot([ps[i][0], ps[j][0]], [ps[i][1], ps[j][1]], 'k')
    print(d)
    fr2 = time.time()
    print(fr2 - fr1)

    plt.figure(1)
    plt.axis('off')
    ps=np.concatenate([ps[:-8],ps[-7:]],axis=0)
    [c, d] = delaunay_net(ps, bigtri=False)
    print(len(d))
    for i in range(ps.shape[0]):
        for j in range(ps.shape[0]):
            if i > j and c[i, j] == 1:
                plt.plot([ps[i][0], ps[j][0]], [ps[i][1], ps[j][1]], 'k')

    e = np.array(d)
    print(e.shape)

    plt.show()
