from numba import jit,njit
import numpy as np
import time


# 三点法求解平面方程
@jit(nopython=True)
def Find_plane_equation(points_face):     # 三点法求解平面方程
    xo1, yo1, zo1 = points_face[0]
    xo2, yo2, zo2 = points_face[1]
    xo3, yo3, zo3 = points_face[2]
    a = (yo2 - yo1) * (zo3 - zo1) - (zo2 - zo1) * (yo3 - yo1)
    b = (xo3 - xo1) * (zo2 - zo1) - (xo2 - xo1) * (zo3 - zo1)
    c = (xo2 - xo1) * (yo3 - yo1) - (xo3 - xo1) * (yo2 - yo1)
    d = -(a * xo1 + b * yo1 + c * zo1)
    Equation_parameters = np.array([a, b, c, d])
    return Equation_parameters

# 求解直线与平面的交点
@jit(nopython=True)
def Find_intersection(p, Equation_parameters):  # 求解直线与平面的交点
    x1, y1, z1 = p
    a, b, c, d = Equation_parameters
    m = x1 - 0.0
    n = y1 - 0.0
    p = z1 - 0.0
    t = (-a * x1 - b * y1 - c * z1 - d) / (a * m + b * n + c * p)
    x = m * t + x1
    y = n * t + y1
    z = p * t + z1
    X = np.array([x, y, z],dtype='float32')
    return X

# 方式一、快
# 判断点p是否在3个points所在的平面上
# @njit
@jit(nopython=True)
def Trangle_inside(points,p):
    v0 = points[1] - points[0]
    v1 = points[2] - points[0]
    v2 = p - points[0]

    dot00 = np.dot(v0,v0)
    dot01 = np.dot(v0,v1)
    dot02 = np.dot(v0,v2)
    dot11 = np.dot(v1,v1)
    dot12 = np.dot(v1,v2)
    inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inverDeno
    if (u<0 or u>1):
        return False
    v = (dot00 * dot12 - dot01 * dot02) * inverDeno
    if(v<0 or v>1):
        return False
    return u+v <= 1

# 方式二、慢
# 判断点p是否在3个points所在的平面上
def TrangleOrArea(points):
    ab = points[1] - points[0]
    ac = points[2] - points[0]
    d = np.cross(ab, ac)
    s = np.linalg.norm(d) / 2
    return s

def Trangle_inside_2(points,p):
    s = TrangleOrArea(points)
    s1 = TrangleOrArea([points[0],points[1],p])
    s2 = TrangleOrArea([points[0],points[2],p])
    s3 = TrangleOrArea([points[2],points[1],p])
    s_sum = s1+s2+s3
    return np.abs(s-s_sum) < 0.001


# 总的判断，是否相交，返回标志，与交点
@jit(nopython=True)
def f_jiao(point_s,points_face):
    Equation_parameters = Find_plane_equation(points_face)  # face的平面方程
    p_jiao = Find_intersection(point_s, Equation_parameters)  # 场景点与face平面的交点
    flag = Trangle_inside(points_face,p_jiao)  # 交点是否在face面内
    return flag,p_jiao



if __name__ == '__main__':
    points_face1 = np.array([[12.218093, -5.317325, -1.4257739],[12.218093, -5.317325, 0.47422612],
                                [12.872038, -4.50865, 0.47422612]], dtype=np.float32)
    point_1 = np.array([24.222, -10.222, -1.858], dtype=np.float32)

    t = time.time()
    for j in range(1024):
        for i in range(200):
            flag,p_jiao = f_jiao(point_1,points_face1)
    print(time.time() - t)

    # p1 = [6.54426157, 2.97825466, -1.38965887]
    # p2 = [6.50719679, 2.95331414, -1.32822887]
    # p3 = [6.53608582, 2.89802243, -1.42364507]
    #
    # p = [(p1[0]+p2[0])/2,(p1[1]+p2[1])/2,(p1[2]+p2[2])/2]
    # pp = [(p[0]+p3[0])/2,(p[1]+p3[1])/2,(p[2]+p3[2])/2]
    # ppp = [pp[0]/2,pp[1]/2,pp[2]/2]
    # pppp = [1.0,1.0,1.0]
    #
    # points_face2 = np.array([p1, p2, p3], dtype=np.float32)
    # point_2 = np.array(ppp, dtype=np.float32)
    # flag,p_jiao = f_jiao(point_2,points_face2)
    # print(flag)
    # print(p_jiao)











