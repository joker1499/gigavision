import json
import math
import os
import subprocess
from time import sleep

import cv2
import numpy as np


def closest_point_2_lines(oa, da, ob,
                          db):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def cam2args():
    for cam in os.listdir("./cams"):
        with open("./cams/" + cam, "r", encoding="utf-8") as f:
            array = f.readlines()
            dict = {}
            dict["fx"] = array[7].split(" ")[0]
            dict["cx"] = array[7].split(" ")[2]
            dict["fy"] = array[8].split(" ")[1]
            dict["cy"] = array[8].split(" ")[2]
            return dict

def main(dict):
    cam_array = []
    file_array = []
    fl_x = float(dict["fx"])
    fl_y = float(dict["fy"])
    cx = float(dict["cx"])
    cy = float(dict["cy"])
    wd = float(cx * 2)
    h = float(cy * 2)
    k1 = 0
    k2 = 0
    p1 = 0
    p2 = 0
    aabb_scale = 16
    angle_x = math.atan(wd / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2

    for file in os.listdir("./images"):
        s1 = file[:8]
        for cam in os.listdir("./cams"):
            if cam[:8] == s1:
                cam_array.append(cam)

    for file in os.listdir("./images"):
        file_path = "./images/" + file
        file_array.append(file_path)

    flag = 0
    out = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "cx": cx,
        "cy": cy,
        "w": wd,
        "h": h,
        "aabb_scale": aabb_scale,
        "frames": [],
    }

    for cam in cam_array:
        with open("./cams/" + cam, "r", encoding="utf-8") as f:
            bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
            up = np.zeros(3)
            w = f.readlines()
            Array = []
            array1 = w[1].split(" ")[:3]
            t1 = w[1].split(" ")[3]
            array2 = w[2].split(" ")[:3]
            t2 = w[2].split(" ")[3]
            # array2.pop()
            array3 = w[3].split(" ")[:3]
            t3 = w[3].split(" ")[3]
            Array.append(tuple(map(float, array1)))
            Array.append(tuple(map(float, array2)))
            Array.append(tuple(map(float, array3)))
            t_array = [float(t1), float(t2), float(t3)]
            R = np.array(Array).reshape([3, 3])
            t = np.array(t_array).reshape([3, 1])
            # print(R)
            # print(t)
            m = np.concatenate([np.concatenate([R, t], 1), bottom],
                               0)  # bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
            c2w = np.linalg.inv(m)
            c2w[0:3, 2] *= -1  # flip the y and z axis
            c2w[0:3, 1] *= -1
            c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
            c2w[2, :] *= -1  # flip whole world upside down

            up += c2w[0:3, 1]  # 第二行
            file_path = file_array[flag]
            flag = flag + 1
            frame = {"file_path": file_path, "sharpness": sharpness(file_path), "transform_matrix": c2w}
            out["frames"].append(frame)
    nframes = len(out["frames"])
    up = up / np.linalg.norm(up)  # 求范数 取模
    print("up vector was", up)  # 矢量
    R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1] 旋转
    R = np.pad(R, [0, 1])  # 填充
    R[-1, -1] = 1  #
    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(R, f["transform_matrix"])  # rotate up to be the z axis

        # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = f["transform_matrix"][0:3, :]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.00001:
                totp += p * w
                totw += w
    if totw > 0.0:
        totp /= totw
    print("***********************************************************")
    print(totp)  # the cameras are looking at totp
    print("****************************************************************")
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] -= totp

    avglen = 0.
    for f in out["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
    avglen /= nframes
    print("***********************************************************")
    print(avglen)
    print("nframes={}".format(nframes))
    print("****************************************************************")

    print("avg camera distance from origin", avglen)
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    print(nframes, "frames")
    with open("transforms.json", "w") as outfile:
        json.dump(out, outfile, indent=2)


def cams2json(dist):
    fl_x = float(dist["fx"])
    fl_y = float(dist["fy"])
    cx = float(dist["cx"])
    cy = float(dist["cy"])
    wd = float(cx) * 2
    h = float(cy) * 2
    k1 = 0
    k2 = 0
    p1 = 0
    p2 = 0
    aabb_scale = 16
    angle_x = math.atan(wd / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2
    out = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "cx": cx,
        "cy": cy,
        "w": wd,
        "h": h,
        "aabb_scale": aabb_scale,
        "frames": [],
    }
    file_array = []
    for file in os.listdir("./cams"):
        file_path = "./images/" + file[:8] + ".jpg"
        file_array.append(file_path)
    flag = 0
    for cam in os.listdir("./cams"):
        with open("./cams/" + cam, "r") as f:
            bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
            up = np.zeros(3)
            w = f.readlines()
            Array = []
            array1 = w[1].split(" ")[:3]
            t1 = w[1].split(" ")[3]
            array2 = w[2].split(" ")[:3]
            t2 = w[2].split(" ")[3]
            # array2.pop()
            array3 = w[3].split(" ")[:3]
            t3 = w[3].split(" ")[3]
            Array.append(tuple(map(float, array1)))
            Array.append(tuple(map(float, array2)))
            Array.append(tuple(map(float, array3)))
            t_array = [float(t1), float(t2), float(t3)]
            R = np.array(Array).reshape([3, 3])
            t = np.array(t_array).reshape([3, 1])
            print(R)
            print(t)
            m = np.concatenate([np.concatenate([R, t], 1), bottom],
                               0)  # bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
            c2w = np.linalg.inv(m)
            c2w[0:3, 2] *= -1  # flip the y and z axis
            c2w[0:3, 1] *= -1
            c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
            c2w[2, :] *= -1  # flip whole world upside down

            up += c2w[0:3, 1]  # 第二行
            file_path = file_array[flag]
            flag = flag + 1
            frame = {"file_path": file_path, "transform_matrix": c2w}
            out["frames"].append(frame)
    nframes = len(out["frames"])
    up = up / np.linalg.norm(up)  # 求范数 取模
    print("up vector was", up)  # 矢量
    R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1] 旋转
    R = np.pad(R, [0, 1])  # 填充
    R[-1, -1] = 1  #
    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(R, f["transform_matrix"])  # rotate up to be the z axis

        # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = f["transform_matrix"][0:3, :]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.00001:
                totp += p * w
                totw += w
    if totw > 0.0:
        totp /= totw
    print("***********************************************************")
    print(totp)  # the cameras are looking at totp
    print("****************************************************************")
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] -= totp

    avglen = 0.
    for f in out["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
    avglen /= nframes
    print("***********************************************************")
    print(avglen)
    print("nframes={}".format(nframes))
    print("****************************************************************")

    print("avg camera distance from origin", avglen)
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    print(nframes, "frames")
    with open("./json/x.json", "w") as outfile:
        json.dump(out, outfile, indent=2)


def distilljson(filename, dist):
    file_array = []
    files = os.listdir("./images")
    with open(filename, 'r', encoding='utf-8') as f:
        jip = f.read()
        aim_json = json.loads(jip)
        flag = 0
        for file in aim_json["frames"]:
            if str(file["file_path"]).split('/')[-1] == files[flag]:
                # print(files[flag])
                flag = flag + 1
                if flag == len(files):
                    flag = flag - 1
            else:
                file_array.append("./images/" + str(file["file_path"]).split('/')[-1])
            # flag = flag + 1

        fl_x = float(dist["fx"])
        fl_y = float(dist["fy"])
        cx = float(dist["cx"])
        cy = float(dist["cy"])
        wd = float(cx) * 2
        h = float(cy) * 2
        k1 = 0
        k2 = 0
        p1 = 0
        p2 = 0
        aabb_scale = 16
        angle_x = math.atan(wd / (fl_x * 2)) * 2
        angle_y = math.atan(h / (fl_y * 2)) * 2
        out = {
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "cx": cx,
            "cy": cy,
            "w": wd,
            "h": h,
            "aabb_scale": aabb_scale,
            "frames": [],
        }
        flag = 0
        # print("dkf" + str(len(aim_json["frames"])))
        for file in aim_json["frames"]:
            if str(file["file_path"]) == file_array[flag]:
                print(file_array[flag])
                out["frames"].append(file)
                flag = flag + 1
                if flag == len(file_array):
                    flag = flag - 1
            else:
                pass

        with open("./json/y.json", "w") as outfile:
            json.dump(out, outfile, indent=2)


def boost(filename, dist):
    with open(filename, 'r', encoding='utf-8') as f:
        jip = f.read()
        aim_json = json.loads(jip)
        fl_x = float(dist["fx"])
        fl_y = float(dist["fy"])
        cx = float(dist["cx"])
        cy = float(dist["cy"])
        wd = float(cx) * 2
        h = float(cy) * 2
        k1 = 0
        k2 = 0
        p1 = 0
        p2 = 0
        aabb_scale = 16
        angle_x = math.atan(wd / (fl_x * 2)) * 2
        angle_y = math.atan(h / (fl_y * 2)) * 2
        out = {
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "cx": cx,
            "cy": cy,
            "w": wd,
            "h": h,
            "aabb_scale": aabb_scale,
            "frames": [],
        }
        # out["frames"].append(aim_json["frames"][0])
        # print(len(aim_json["frames"]))
        for i in range(len(aim_json["frames"])):
            out["frames"].append(aim_json["frames"][i])
            if (i + 1) % 2 == 0 or i + 1 == len(aim_json["frames"]):
                with open("./json/z{}.json".format(i), "w") as outfile:
                    json.dump(out, outfile, indent=2)
                out["frames"] = []


if __name__ == '__main__':

    dist = cam2args()
    main(dist)
    path_name = os.path.abspath(os.curdir).split("\\")[-1]
    s = r"python ../../../scripts/run.py --mode nerf --scene ../../../data/nerf/{} --save_snapshot  " \
        r"../../../data/nerf/{}/base.msgpack".format(path_name,path_name)
    p = subprocess.Popen(s)
    sleep(320)
    p.kill()
    if not os.path.exists('./json'):
        os.mkdir(r'./json')
    cams2json(dist)
    distilljson(r"./json/x.json",dist)
    boost("./json/y.json", dist)
    json_names = os.listdir("./json")
    json_names.pop(0)
    json_names.pop(0)
    print(json_names)
    for json_name in json_names:
        s1 = r"python ../../../scripts/run.py --scene ../../../data/nerf/{} --mode nerf --load_snapshot " \
             r"../../../data/nerf/{}/base.msgpack --screenshot_transforms ../../../data/nerf/{}/json/{} --screenshot_dir " \
             r"../../../data/nerf/{}/screenshot --width {} --height {} --n_steps 0".format(path_name,path_name,path_name,
                                                                                           json_name,path_name,
                                                                                           int(dist["cx"][:4]),int(dist["cy"][:4]))
        p = subprocess.Popen(s1)
        sleep(300)
        p.kill()

