
import sys
sys.path.append(r'/code/daima/16.mixamo/14、分割_pe_database/utils')
from scipy.spatial import distance
import numpy as np
from sklearn.preprocessing import StandardScaler  # 规范化
from file_read_write import load_velodyne_points
from scipy.stats import wasserstein_distance
import copy


def distance_methods(x,y):

    dis = distance.braycurtis(x, y)
    print('布雷柯蒂斯相异度（Bray-Curtis distance）',dis)

    dis = distance.canberra(x, y)
    print('坎贝拉距离（Canberra distance）',dis)

    dis = distance.chebyshev(x, y)
    print('切比雪夫距离（Chebyshev distance）',dis)

    dis = distance.cityblock(x, y)
    print('曼哈顿距离（Manhattan／cityblock distance）',dis)

    dis = distance.correlation(x, y)
    print('相关系数距离（Correlation distance）',dis)

    dis = distance.cosine(x, y)
    print('余弦相似度距离（Cosine distance）',dis)

    dis = distance.euclidean(x, y)
    print('欧氏距离（Euclidean distance）',dis)

    from scipy import stats as sts
    def JS_divergence(p, q):
        M = (p + q) / 2
        return 0.5 * sts.entropy(p, M) + 0.5 * sts.entropy(q, M)
    dis = JS_divergence(x, y)
    print('JS散度距离（Jensen-Shannon distance）',dis)

    dis = distance.sqeuclidean(x, y)
    print('平方欧式距离（squared Euclidean distance）',dis)

    dis = distance.wminkowski(x, y, 2, np.ones(3))
    print('加权闵可夫斯基距离（Minkowski distance）',dis)


    from scipy.stats import wasserstein_distance
    dis = wasserstein_distance(x, y)
    print('1搬土距离（Earth Mover distance）',dis)

    # x = [0.5, 3.3881989165193145e-06, 0.007009673349221927, 2.7785622593068027, 2.7785622593068027, 1.0,
    #      0.1480135071948422, 2.7785622593068027, 2, 0.0, 0.0, 0.02111525564774837, 0, 0, 3.3881989165193145e-06,
    #      0.02111525564774837, 1.0, 0.02111525564774837, 0.28901734104046245, 0.0, 0, 0.0, 0.0, 1, 0.02111525564774837,
    #      0.0, 3.3881989165193145e-06]
    # y = [0.8, 6.859405279689656, 0.0037439161362785474, 4020.4096644631295, 0.005439330543933054, 0.08928571428571429,
    #      0.04654587589796659, 128609.0, 5, 0.7678571428571429, 0.03798619846624095, 0.24815204448802128,
    #      -0.017954805269944772, 0, 358.62096982747676, 13.421226391252906, -0.017857142857142856, -8.571428571428571,
    #      0.1179245283018868, 0.028545153041402063, 0.06847760995576437, 0.5714285714285714, 0.0, 112,
    #      358.62096982747676, 64.26004935863212, -1.2244897959183674]
    # dis = wasserstein_distance(x, y)
    # print('2搬土距离（Earth Mover distance）', dis)
    #
    # x = [3.4, 3.9, 7.5, 7.8]
    # x_w = [1.4, 0.9, 3.1, 7.2]
    # y = [4.5, 1.4]
    # y_w = [3.2, 3.5]
    # dis = wasserstein_distance(x, y, x_w, y_w)
    # print('3搬土距离（Earth Mover distance）', dis)
    #
    # x = [0.0]
    # x_w = [10.0]
    # y = [0.0]
    # y_w = [2.0]
    # dis = wasserstein_distance(x, y, x_w, y_w)
    # print('4搬土距离（Earth Mover distance）', dis)


'''
相似满足的条件
1、点，平移，旋转，值不变
2、
3、
'''

a = np.array([[2104, 3],[1600, 3],[2400, 3],[1416, 2],[3000, 4],[1985, 4]])
x = np.asarray([[1,1,1],[1,1,1],[1,1,1],])
y = np.asarray([[2,2,2],[2,2,2],[2,2,2],])

c = np.asarray([1,2,3,4,5])
d = np.asarray([1,3,5])

# distance_methods(x,y)

points1_path = '/code/daima/16.mixamo/14、分割_pe_database/data/similar_data/lidar_points_box_remove.bin'
points2_path = '/code/daima/16.mixamo/14、分割_pe_database/data/similar_data/pe_points_box.bin'

points1 = load_velodyne_points(points1_path)
points2 = load_velodyne_points(points2_path)

points1_model = StandardScaler()
points2_model = StandardScaler()

points1 = points1_model.fit_transform(points1)
points2 = points2_model.fit_transform(points2)

dis = wasserstein_distance(points1[:,0], points2[:,0]) + \
      wasserstein_distance(points1[:,1], points2[:,1]) + \
      wasserstein_distance(points1[:,2], points2[:,2]) + \
      wasserstein_distance(points1[:,3], points2[:,3])

print('1搬土距离（Earth Mover distance）',dis)












# print('缩放后的结果:')
# print(x)
# print('每个特征的缩放比例:{}'.format(model.scale_))
# print('每个特征的均值:{}'.format(model.mean_))
# print('每个特征的方差:{}'.format(model.var_))
