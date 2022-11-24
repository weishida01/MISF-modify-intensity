import numpy as np
import get_calib
import pymeshlab

# 2、在screen，初始选取points位置点，作为插入人的位置  返回points
def points_generate(screen,calib_path):
    screen = screen
    sc = screen.current_mesh()
    sc_min = sc.bounding_box().min()  # 人点云包围盒的最小值
    screen_x_length = sc.bounding_box().dim_x()  # 人点云包围盒的x长度
    screen_y_length = sc.bounding_box().dim_y()  # 人点云包围盒的y长度

    box_bum = 20  # 候选框的维度，个数为 （box_bum+1）* （box_bum+1）
    step_length = 1  # 候选框移动的步长

    calib = get_calib.Calibration(calib_path)

    # 候选的框的坐标
    points = []
    # 只在相机视角的正前方生成候选区域
    for i in range(0, box_bum * 2):
        for j in range(-box_bum, box_bum):
            if i*step_length < (sc_min[0]+screen_x_length) and i*step_length > sc_min[0]:
                if j*step_length < (sc_min[1]+screen_y_length) and j*step_length > sc_min[1]:
                    point = np.array([i * step_length, j * step_length])
                    v = np.array([[point[0], point[1], 0]])
                    pts_img , pts_depth= calib.lidar_to_img(v)
                    cam_location = pts_img[0]
                    if cam_location[0] > 20 and cam_location[0] < 1200:
                        if cam_location[1] > 20 and cam_location[1] < 360:
                            points.append(point)
    return points



# 3、筛选screen，中符合要求的points位置点（可以放置人的位置）
def points_select(points,screen_path):
    screen = pymeshlab.MeshSet()
    screen.load_new_mesh(screen_path)

    sc = screen.current_mesh()
    screen_location_min = sc.bounding_box().min()  # 场景点云包围盒的最小值
    screen_z_length = sc.bounding_box().dim_z()  # 人点云包围盒的z长度

    pe_box_x_length = 0.6  # 筛选的box的x大小
    pe_box_y_length = 0.6  # 筛选的box的y大小
    pe_box_z_length = 2.0  # 筛选的box的z大小

    points_num_min = 10  # 点筛选最少的个数
    points_max_min = 0.2  # 点筛选，点最大z值与最小z的差值
    points_mean = screen_z_length + screen_location_min[2] - pe_box_z_length  # 点筛选，点的高度小于参考值

    # 判断点是否合适
    final_points = []
    for i in range(len(points)):
        curb_p = points[i]
        curb_x_length = pe_box_x_length/2
        curb_y_length = pe_box_y_length/2

        # 选中curb中的点
        str_select = "(x>{}&&x<{}&&y>{}&&y<{})".format(curb_p[0]-curb_x_length,curb_p[0]+curb_x_length,curb_p[1]-curb_y_length,curb_p[1]+curb_y_length)
        screen.apply_filter('compute_selection_by_condition_per_vertex',condselect=str_select)

        # 点的数量大于points_num_min
        if(screen.current_mesh().selected_vertex_number()>points_num_min):
            select_points = []
            for index,flag in enumerate(screen.current_mesh().vertex_selection_array()):
                if flag == True:
                    select_points.append(screen.current_mesh().vertex_matrix()[index])
            select_points_x = []
            for select_point in select_points:
                select_points_x.append(select_point[0])
            select_points_y = []
            for select_point in select_points:
                select_points_y.append(select_point[1])
            select_points_z = []
            for select_point in select_points:
                select_points_z.append(select_point[2])

            # 点的最大值最小值差在一定范围，为了是一个平面
            if max(select_points_z) - min(select_points_z) < points_max_min:
                # 点的均值不要太高，让平面上放有人的空间
                if sum(select_points_z)/len(select_points_z) < points_mean:
                    # 为了点不是太聚集，是一个平面
                    if max(select_points_x) - min(select_points_x) > (curb_x_length * 1.5):
                        if max(select_points_y) - min(select_points_y) > (curb_y_length * 1.5):
                            final_points.append(np.array([curb_p[0],curb_p[1],max(select_points_z)]))
                            # print("条件满足点：points_location,index:",i,"curb_p:",curb_p)
    return final_points