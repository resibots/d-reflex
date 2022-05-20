# -*- coding: utf-8 -*-
# # config

# ## Importations

# +
import numpy as np
import os
import pickle 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA 
import subprocess
from datetime import datetime
import random 
import yaml 
import itertools
from tqdm import tqdm
from time import time, sleep
from copy import copy, deepcopy
import multiprocessing as mp
from scipy.stats import ttest_ind, ttest_rel, shapiro, mannwhitneyu, pearsonr
from shutil import copyfile
import pylab
from colorama import Fore
from IPython.display import HTML
import seaborn as sb
import pandas as pd
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
from ipywidgets import interact
from subprocess import CalledProcessError
from torch import nn
from torch.autograd import Variable
import torch
import queue
from numpy.random import default_rng 
import scipy.signal
from PIL import Image
import cv2  # opencv-python

rng = default_rng()
# -

# ## Matplotlib

# +
font = {'size'   : 18}
mpl.rc('font', **font)

plt.rcParams["figure.figsize"] = (16, 9)

blue = '#332288'
green = '#117733'
light_green = '#44AA99'
light_blue = '#88CCEE'
yellow = '#DDCC77'
red = "#CC6677"
grad = [blue, light_blue, light_green, green, yellow, red, '#AA4499']
colors = [light_green, red, light_blue, green, blue, yellow]
# -

# ## Talos 

# ![alt text](../../Kinematics-of-TALOS.png "Title")

# ### Joints description

# +
joints = [[0,'rootJoint_pos_x',-np.inf,np.inf],
[1,'rootJoint_pos_y',-np.inf,np.inf],
[2,'rootJoint_pos_z',-np.inf,np.inf],
[3,'rootJoint_rot_x',-np.inf,np.inf],
[4,'rootJoint_rot_y',-np.inf,np.inf],
[5,'rootJoint_rot_z',-np.inf,np.inf],
[6,'leg_left_1_joint',-0.349066,1.5708],
[7,'leg_left_2_joint',-0.5236,0.5236],
[8,'leg_left_3_joint',-2.095,0.7],
[9,'leg_left_4_joint',0,2.618],
[10,'leg_left_5_joint',-1.27,0.68],
[11,'leg_left_6_joint',-0.5236,0.5236],
[12,'leg_right_1_joint',-1.5708,0.349066],
[13,'leg_right_2_joint',-0.5236,0.5236],
[14,'leg_right_3_joint',-2.095,0.7],
[15,'leg_right_4_joint',0,2.618],
[16,'leg_right_5_joint',-1.27,0.68],
[17,'leg_right_6_joint',-0.5236,0.5236],
[18,'torso_1_joint',-1.25664,1.25664],
[19,'torso_2_joint',-0.226893,0.733038],
[20,'arm_left_1_joint',-1.5708,0.785398],
[21,'arm_left_2_joint',0.00872665,2.87107],
[22,'arm_left_3_joint',-2.42601,2.42601],
[23,'arm_left_4_joint',-2.23402,-0.00349066],
[24,'arm_left_5_joint',-2.51327,2.51327],
[25,'arm_left_6_joint',-1.37008,1.37008],
[26,'arm_left_7_joint',-0.680678,0.680678],
[27,'gripper_left_inner_double_joint',-1.0472,0],
[28,'gripper_left_fingertip_1_joint',0,1.0472],
[29,'gripper_left_fingertip_2_joint',0,1.0472],
[30,'gripper_left_inner_single_joint',0,1.0472],
[31,'gripper_left_fingertip_3_joint',0,1.0472],
[32,'gripper_left_joint',-0.959931,0],
[33,'gripper_left_motor_single_joint',0,1.0472],
[34,'arm_right_1_joint',-0.785398,1.5708],
[35,'arm_right_2_joint',-2.87107,-0.00872665],
[36,'arm_right_3_joint',-2.42601,2.42601],
[37,'arm_right_4_joint',-2.23402,-0.00349066],
[38,'arm_right_5_joint',-2.51327,2.51327],
[39,'arm_right_6_joint',-1.37008,1.37008],
[40,'arm_right_7_joint',-0.680678,0.680678],
[41,'gripper_right_inner_double_joint',-1.0472,0],
[42,'gripper_right_fingertip_1_joint',0,1.0472],
[43,'gripper_right_fingertip_2_joint',0,1.0472],
[44,'gripper_right_inner_single_joint',0,1.0472],
[45,'gripper_right_fingertip_3_joint',0,1.0472],
[46,'gripper_right_joint',-0.959931,0],
[47,'gripper_right_motor_single_joint',0,1.0472],
[48,'head_1_joint',-0.20944,0.785398],
[49,'head_2_joint',-1.309,1.309]
]
lower_limits = []
upper_limits = []
for i in range(50):
    if i<3:
        lower_limits.append(0)
        upper_limits.append(1)
    elif i<6:
        lower_limits.append(-np.pi)
        upper_limits.append(np.pi)
    else:
        lower_limits.append(joints[i][2])
        upper_limits.append(joints[i][3])
lower_limits = np.array(lower_limits)
upper_limits = np.array(upper_limits)

talos_body = {
    "left_leg": [6,7,8,9,10,11],
    "right_leg": [12,13,14,15,16,17],
    "torso": [18,19],
    "left_arm": [20,21,22,23,24,25,26],
    "right_arm": [34,35,36,37,38,39,40],
    "head": [48,49],
}

talos_grouped_joints = {
    "l_hip": [6,7,8], 
    "r_hip": [12,13,14],
    "l_knee": [9],
    "r_knee": [15],
    "l_ankle": [10,11],
    "r_ankle": [16,17],
    "torso": [18,19],
    "l_shoulder": [20,21,22],
    "r_shoulder": [34,35,36],
    "l_elbow": [23,24],
    "r_elbow": [37,38],
}

def input_normalization(x):
    return (x-lower_limits)/(upper_limits-lower_limits)

discrepancy_joints_names  = ["leg_left_1_joint", "leg_left_2_joint", "leg_left_3_joint", "leg_left_4_joint", 
                             "leg_left_5_joint", "leg_left_6_joint", "leg_right_1_joint", "leg_right_2_joint", 
                             "leg_right_3_joint", "leg_right_4_joint", "leg_right_5_joint", "leg_right_6_joint",
                             "torso_1_joint", "torso_2_joint", "arm_left_1_joint", "arm_left_2_joint", 
                             "arm_left_3_joint", "arm_left_4_joint", "arm_right_1_joint", "arm_right_2_joint",
                             "arm_right_3_joint", "arm_right_4_joint"]
index32_to_50 = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                 22, 23, 24, 25, 26, 32, 34, 35, 36, 37, 38, 39, 40, 46, 48, 49]
names32 = [joints[i][1] for i in index32_to_50 ]

# +
short = [
 'L_Leg_1',
 'L_Leg_2',
 'L_Leg_3',
 'L_Leg_4',
 'L_Leg_5',
 'L_Leg_6',
 'R_Leg_1',
 'R_Leg_2',
 'R_Leg_3',
 'R_Leg_4',
 'R_Leg_5',
 'R_Leg_6',
 'Torso_1',
 'Torso_2',
 'L_Arm_1',
 'L_Arm_2',
 'L_Arm_3',
 'L_Arm_4',
 'L_Arm_5',
 'L_Arm_6',
 'L_Arm_7',
 'R_Arm_1',
 'R_Arm_2',
 'R_Arm_3',
 'R_Arm_4',
 'R_Arm_5',
 'R_Arm_6',
 'R_Arm_7',
]
    
long = [
 'leg_left_1_joint',
 'leg_left_2_joint',
 'leg_left_3_joint',
 'leg_left_4_joint',
 'leg_left_5_joint',
 'leg_left_6_joint',
 'leg_right_1_joint',
 'leg_right_2_joint',
 'leg_right_3_joint',
 'leg_right_4_joint',
 'leg_right_5_joint',
 'leg_right_6_joint',
 'torso_1_joint',
 'torso_2_joint',
 'arm_left_1_joint',
 'arm_left_2_joint',
 'arm_left_3_joint',
 'arm_left_4_joint',
 'arm_left_5_joint',
 'arm_left_6_joint',
 'arm_left_7_joint',
 'arm_right_1_joint',
 'arm_right_2_joint',
 'arm_right_3_joint',
 'arm_right_4_joint',
 'arm_right_5_joint',
 'arm_right_6_joint',
 'arm_right_7_joint',
]

short_to_long = {short[i]: long[i] for i in range(len(short))}
# -

# ## Conditions

# +
Conditions_raw = {
    "Nothing": {"Default": None},

    "Collision": {
        "Sphere_wrist": [True,[-0.05,  0.45, 0.95, 0., 0., 0., 0.2, 0.2, 0.2]],
        "Sphere_elbow": [True, [-0.23,  0.45, 1.25, 0., 0., 0., 0.2, 0.2, 0.2]],
        "Sphere_hand": [True, [0.15, 0.55, 1., 0., 0., 0., 0.2, 0.2, 0.2]],
        "Vertical_bar":  [False,[0.05, 0.55, 1., 0., 0., 0., 0.1, 0.1, 1.]],
        "Horizontal_bar": [False, [0.1, 0.55, 1.05, 0., 0., 0., 1, 0.1, 0.1]],
        "Cube_wrist":[False, [-0.05,  0.5, 0.9, 0., 0., 0., 0.2, 0.2, 0.2]],
        "Cube_elbow": [False, [0.1,  0.45, 1.25, 0., -0.7, 0., 0.175, 0.175, 0.175]],
        "Cube_hand": [False, [0.16, 0.57, 1., 0., 0., 0., 0.175, 0.175, 0.175]],
        "Rotated_cube": [False, [0.3, 1.02, 0.7 ,0.78, 0, 0.78, 0.2, 0.2, 0.2]],
        "Flat_cube": [False, [0.18, 0.25, 1.1, -0.75, 0, 0., 0.2, 0.3, 0.4]],
        "Wall": [False, [-0.1, 0.56, 1.1, 0., 0., 0., 1, 0.05, 1]],
    },

    "Locked": deepcopy(short_to_long),

    "Passive": deepcopy(short_to_long),
    
    "Weak": deepcopy(short_to_long),
    
    "Cut":  {key: long.replace("joint", "link") for (key,long) in short_to_long.items() if "torso" not in long},
}

collision_acronymes = {
                        "Sphere_wrist": "SW", "Sphere_elbow": "SE", "Sphere_hand": "SH", "Vertical_bar": "VB",
                        "Horizontal_bar": "HB", "Cube_wrist": "CW","Cube_elbow": "CE","Cube_hand": "CH", 
                        "Rotated_cube": "RC","Flat_cube": "FC", "Wall": 'WA', 
                       }

Conditions = {}           
yaml_types = {"Passive": 'passive', "Collision": 'colision_shapes', "Locked": 'locked', "Weak": 'weak_joints', "Cut": "amputated", "Nothing": None}    

for condition_type in Conditions_raw:
    conditions = Conditions_raw[condition_type]
    conditions_dict = {}
    for i, key in enumerate(conditions):
        if condition_type == "Collision": 
            acronyme = collision_acronymes[key]
        elif condition_type == "Default":
            acronyme = "DE"
        else:
            acronyme = ("L" if condition_type == "Locked" else "P") + str(i)
        full_name = condition_type + "_" + key
        if condition_type == "Weak":
            conditions_dict[full_name] = {"full name": full_name, "type":yaml_types[condition_type], "name": key, 
                                          "joints": conditions[key], "acronyme": acronyme, "weakness": 0.}
        else:
            conditions_dict[full_name] = {"full name": full_name, "type":yaml_types[condition_type], "name": key, 
                                          "joints": conditions[key], "acronyme": acronyme}
    Conditions[condition_type[0]] = conditions_dict

Conditions["A"] = {**Conditions["N"], **Conditions["L"],  **Conditions["P"], **Conditions["W"], **Conditions["C"]}
Conditions["D"] = {**Conditions["N"], **Conditions["L"], **Conditions["W"]}
for key in ["A", "D", "L", "P", "C", "N", "W"]:
    Conditions[f"{key}_names"] = list(Conditions[key].keys())

def one_condition(key):
    return {key: Conditions["A"][key]}


# -

# # Fonctions Definitions 

# ### create_xp_folder, set_yaml, execute

# +
def create_xp_folder(stamp_id=None):
    now = datetime.now()
    if stamp_id is None:
        timestamp = now.strftime("%Y/%m/%d/%H:%M:%S")+"_{:4.0f}".format((random.randint(1000,10000)))
    else:
        timestamp = now.strftime("%Y/%m/%d/%H:%M:%S")+"_{:4.0f}".format((random.randint(1000,10000)))+str(stamp_id)
    xp_folder = "/home/pal/humanoid_adaptation/data/"+timestamp
    return xp_folder

def set_yaml(yaml_path, yaml_name, run_folder, config, subtype):
    with open(yaml_path+yaml_name, 'r') as f:
        doc = yaml.load(f, Loader=yaml.FullLoader)
    if subtype in ["TASK"]:
        if "torso" in doc:
            doc["torso"]["mask"] = "000110"
        if "momentum" in doc:
            doc["momentum"]["mask"] = "000110"
        for key, val in config[subtype].items():
            if key in doc:
                for key2, val2 in val.items():
                        doc[key][key2] = val2
            else:
                print(f"WARNING:{key} is not in yaml original file")
    else:
        for key, val in config[subtype].items():
            if key in doc[subtype]:
                if type(val) == dict:
                    for key2, val2 in val.items():
                        doc[subtype][key][key2] = val2
                else:
                    doc[subtype][key] = val
            else:
                print(f"WARNING:{key} is not in yaml original file")

    with open(run_folder + "/" + yaml_name, 'w') as f:
        yaml.dump(doc, f)

def execute(config, run_folder, actuator="spd", verbose=0, with_video=False, with_recording=False, with_ghost=False, fast=False):
    assert(actuator in ["torque", "spd", "servo"])
    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(run_folder+"/behaviors", exist_ok=True)
    os.makedirs(run_folder+"/stabilizer", exist_ok=True)
    with open(run_folder+"/config.pk", 'wb') as f:
        pickle.dump(config, f)
        
    yaml_path = "/home/pal/humanoid_adaptation/etc/"
    controller_name = "damage_controller.yaml"
    behavior = "hands.yaml"
    # stabilizer
    for stab in ["double_support.yaml", "fixed_base.yaml", "single_support.yaml"]:
        copyfile(yaml_path+"stabilizer/"+stab, run_folder+"/stabilizer/"+stab)
    # controller 
    set_yaml(yaml_path, controller_name, run_folder, config, "CONTROLLER")
    # behavior 
    set_yaml(yaml_path+"behaviors/","hands.yaml", run_folder+"/behaviors", config, "BEHAVIOR")
    # tasks
    set_yaml(yaml_path, "tasks.yaml", run_folder, config, "TASK")
    for file in ["configurations.srdf", "collision_thresholds.yaml", "frames.yaml"]:
        copyfile(yaml_path+file, run_folder+"/"+file)

    if with_video or with_recording:
        exe = "/home/pal/humanoid_adaptation/build/damage_run_graphics"
    else:
        exe = "/home/pal/humanoid_adaptation/build/damage_run"
    conf = [exe, "-c", run_folder + "/" + controller_name, "-b", run_folder+"/behaviors/hands.yaml", "-a", actuator]
    if with_video:
        conf.append("-woff")
    else:
        conf.append("-won")
    if with_ghost:
        conf.append("-g")
    if actuator in ["torque", "servo"]:
        conf.append("--closed_loop")
    if with_recording:
        conf.append("-mvideo.mp4")
    if fast:
        conf.append("-f")
        conf.append("-k")
        conf.append("dart")
    t1 = time()
    if verbose > 2:
        txt = subprocess.check_output(conf).decode()
        if "Talos pos tracker initialized" in txt:
            print(txt.split("Talos Damage Controller initialized")[-1])
        else:
            print(txt)
    else:
        subprocess.check_output(conf)
    return time()-t1


# -

# ### gather_data

def gather_data(folder_path, files):
    data = {}
    for file in files:
        file_name = file.split(".")[0]
        if ".pk" in file:
            with open(folder_path + file, 'rb') as f:
                data[file_name] = pickle.load(f)
        else:
            try:
                data[file_name] = np.loadtxt(folder_path + file)
            except OSError:
                pass 
            except ValueError:
                data[file_name] = None 
    return data


# ### median filter, compute_*, detect_collision, falling

# +
def median_filter(L, k):
    filtered = []
    for i in range(len(L)):
        filtered.append( np.median(L[max(0,i-k):i+1], axis=0))
    return filtered

def median_filter2(L, k):
    filtered = []
    for i in range(len(L)):
        filtered.append( np.median(L[max(0,i-k//2):min(len(L), i+k//2)], axis=0))
    return filtered

def compute_discrepancy(data, real='sensor_tau', tsid='tsid_tau'):
    n = min(len(data[real]), len(data[tsid]))
    return np.linalg.norm(median_filter2(data[real][:n]-data[tsid][:n],10), axis=1)

def compute_foot_discrepancy(data, real='sensor_tau', tsid='tsid_tau'):
    assert('real' in real)
    n = min(len(data[real]), len(data[tsid]))
    return np.abs(median_filter2(np.abs(data[real][:n, 2])-np.abs(data[tsid][:n]), 10))

def compute_tracking_error(data, func=np.mean):
    n = min(len(data['rh_real']),len(data['rh_ref']))
    return func(np.linalg.norm(data['rh_real'][:n]-data['rh_ref'][:n], axis=1))

def compute_max_discrepancy(data):
    n = min(len(data['sensor_tau']),len(data['tsid_tau']))
    return np.max(np.linalg.norm(median_filter2(data['tsid_tau'][:n]-data['sensor_tau'][:n], 10), axis=1)[200:])

def falling(data):
    imu= median_filter(data['imu'][:,2],20)
    i= 100
    while i <len(imu) and imu[i]<-8.45:
        i+=1
    return i


# -

# ### Read_stop_reason, run_online

# +
def extract_body_part_in_contact(data):
    if 'contact_pos' in data:
        if len(data['contact_pos'].shape) == 2:
            return int(data['contact_pos'][0][-1])
        elif len(data['contact_pos'].shape) == 1:
            return int(data['contact_pos'][-1])
        else:
            return None 
    else:
        return None
    
def extract_contact_time(data):
    if 'contact_pos' in data:
        if len(data['contact_pos'].shape) == 2:
            return float(data['contact_pos'][0][0])
        elif len(data['contact_pos'].shape) == 1:
            return float(data['contact_pos'][0])
        else:
            return None 
    else:
        return None
    
def read_stop_reason(data):
    reasons = ["Error", "Falling", "Running", "Fallen_floor", "Fallen_wall", 
               "Unfallen", "Recovered", "Timeout"]
    if "end" in data:
        try:
            end = data['end']
        except KeyError:
            return "Error", 0
        if end is None:
            return "Error", 0
        if len(end) == 0:
            raise NameError("Error empty file")
        i = int(end[1])
        if i>= len(reasons):
            raise NameError("Error not registered reason")
        else:
            reason = reasons[i]
            if reason == "Recovered" and "contact_pos" in data:
                body = extract_body_part_in_contact(data)
                if body is not None:
                    reason = reason + str(body)
            return reason, end[0]
    else:
        return "Error", 0
    
    
def WrongConditionERROR(Exception):
    pass

def run_online(dic, conditions, verbose=0):
    myconfig = dic["config"]
    video = myconfig["arg"]["video"]
    recording = myconfig["arg"]["recording"]
    ghost =  myconfig["arg"]["ghost"]
    condition_names = ""
    for condition in conditions:
        if condition_names != "":
            condition_names += "&"
        condition_names += condition["full name"]
        if condition["type"] is not None and condition["type"] in "weak_joints":
            condition_names += f'_{condition["weakness"]}'
    xp_folder = dic["folder"] + "/" + dic["name"] + '/' + condition_names +"/"
    myconfig["CONTROLLER"]["base_path"] += xp_folder.split("humanoid_adaptation")[1] 
    myconfig["CONTROLLER"]["xp_folder"] = xp_folder
    
    config = [
        ("CONTROLLER", "tasks", "tasks.yaml"),
        ('BEHAVIOR', "name", "hands"),
    ]
    for condition in conditions:
        if condition["type"] in ['passive', 'colision_shapes', 'locked']:
            myconfig['CONTROLLER'][condition["type"]].append(condition["joints"])
        if condition["type"] in ['amputated']:
            myconfig['CONTROLLER'][condition["type"]].append(condition["joints"])
        if condition["type"] == "weak_joints":
            myconfig['CONTROLLER']["weak_joints"].append(condition["joints"])
            myconfig['CONTROLLER']["weakness"].append(condition["weakness"])
    walltime = execute(
        config=myconfig, 
        run_folder=xp_folder, 
        actuator=myconfig["arg"]["actuator"],
        with_video=video, 
        with_recording=recording, 
        with_ghost=ghost,
        verbose=verbose
    )
    data = gather_data(xp_folder, ['end.dat'])
    stop_reason, behavior_time = read_stop_reason(data)
    if verbose > 1:
        print(f"{dic['name']}: time {behavior_time}s (walltime {walltime:1.1f}s) {stop_reason}")
    res = {"stop_reason": stop_reason, "walltime": walltime, "time": behavior_time}
    return (dic["name"], condition_names, res)


# -

# ### make_jobs, worker, master

# +
def make_jobs(dicts, conditions, verbose=0):
    def jobs():
        for dic in dicts.values():
            for condition in conditions:
                yield (run_online, (dic, condition, verbose))
    return jobs(), len(dicts)*len(conditions)

def make_jobs_custom_condition(dicts, verbose=0):
    def jobs():
        for dic in dicts.values():
            yield (run_online, (dic, dic["condition"], verbose))
    return jobs(), len(dicts)

def make_jobs_custom_conditions(dicts, verbose=0):
    count = 0 
    def jobs():
        for dic in dicts.values():
            for condition in dic["conditions"]:
                yield (run_online, (dic, condition, verbose))
    return jobs(), np.sum([len(dic["conditions"]) for dic in dicts.values()])

def worker(job_queue, res_queue):
    while True:
        job = job_queue.get()
        if job == "Done":
            break
        else:
            f, arg = job
            res_queue.put(f(*arg))

def master(dicts, jobs, n_jobs, n_processes=50, verbose=1):
    job_queue = mp.Queue()
    res_queue = mp.Queue()
    n_processes = min(n_processes, n_jobs)
    pool = mp.Pool(n_processes, worker, (job_queue, res_queue))
    xp_folder = create_xp_folder()
    if verbose:
        print(xp_folder)
    for job in jobs:
        (_, (dic, _, _)) = job
        dic["folder"] = xp_folder
        job_queue.put(job)

    for _ in range(n_processes):
        job_queue.put("Done")
    
    todos= []
    for _ in (tqdm(range(n_jobs), smoothing=0.) if verbose else range(n_jobs)):
        try:
            out = res_queue.get(timeout=60)
            todos.append(out)
        except queue.Empty:
            print("There's a timeout job")
    for (name, condition, res) in todos:
        dicts[name]["folder"] = xp_folder+"/"+name
        for key, item in res.items():
            dicts[name][key][condition] = item
        
    pool.terminate()
    return xp_folder


# -

# ### Load // 

# +
def loader(path, name, files=['end']):
    stop_reasons = {}
    xp = path+"/"+name
    res =  {}
    conditions = os.listdir(xp)
    if "config" in files:
        with open(xp+"/"+conditions[0]+"/config.pk", "rb") as f:
            config = pickle.load(f)
        res["config"] = config
        files.remove("config")
    data = {}
    for condition in conditions:
        data[condition] = gather_data(xp+"/"+condition+"/", files)
        if "end.dat" in files:
            stop_reasons[condition] = read_stop_reason(data[condition])
    res["data"] = data
    if "end.dat" in files:
        res["stop_reason"] = stop_reasons
    return name, res

def load_worker(job_queue, res_queue):
    while True:
        job = job_queue.get()
        if job == "Done":
            break
        else:
            f, arg = job
            res_queue.put(f(**arg))
            
def load_master(dicts, jobs, n_jobs, n_processes=50, verbose=1):
    job_queue = mp.Queue()
    res_queue = mp.Queue()
    n_processes = min(n_processes, n_jobs)
    pool = mp.Pool(n_processes, load_worker, (job_queue, res_queue))

    for job in jobs:
        job_queue.put(job)

    for _ in range(n_processes):
        job_queue.put("Done")
    
    todos = []

    for _ in (tqdm(range(n_jobs)) if verbose else range(n_jobs)):
        todos.append(res_queue.get())
    
    for (name, dic) in todos:
        dicts[name] = dic 
    pool.close()
    pool.join()


# -

# # Xp running functions

# ## default parameters

default_params = {
    "CONTROLLER": { 
        "stabilizer": {"activated": False},    
        "base_path": "/home/pal/humanoid_adaptation",
        "closed_loop": False,
        "xp_folder": "",
        "duration": 4.,
        "use_falling_early_stopping": True,
        "fallen_treshold": 0.4,
        "colision_shapes": [[False, [-0.1, -0.9, 1.1, 0., 0., 0., 2, 0.05, 2.]]],
        "damage_time": 0.,
        "locked": [],
        "passive": [],
        "amputated": [],
        "reflex_time": 0.1,
        "use_baseline": False,
        "use_reflex_trajectory": False,
        "update_contact_when_detected": False,
        "reflex_arm_stiffness": 10_000.,
        "reflex_stiffness_alpha": 0.,
        "wall_opacity": [0., 96./255., 1., 0.7],
        "cam_pos":  [3.5, 2, 2.2, 0., 0., 1.4],
    }, 
    "BEHAVIOR": {
        #"com_trajectory_duration": 0.25,
        #"time_before_moving_com": 0.,
        #"com_ratio": 0,
        #"z": 0.0,
        "rh_shift": [0.,0.,0.,0.,0.,0.],
        "lh_shift": [0.,0.,0.,0.,0.,0.],
        "trajectory_duration": 1,
        "reflex_contact_pos": [0., 0., 0.],
        "reflex_speed": 1.,
    },

    "arg": {
        "actuator": "spd",
        "video": False,
        "recording": False,
        "ghost": False,
    },
    "TASK": {
        "com": {
            "weight": 1000.,
        },
        "posture": {
            "weight": 0.3,
        },
        "rh": {
            "weight": 0.,
        },
        "lh": {
            "weight": 0.,
        },
        "torso": {
            "weight": 0.,
        },
        "contact_rhand": {
            "x": 0.,
            "y": 0.,
            "z": 0.,
        }
    }
}


# ## Functions

def test_default(name, actuator="spd", recording=False, video=False, myconfig={}):
    config = deepcopy(default_params)
    config["arg"]["actuator"] = actuator
    config["arg"]["recording"] = recording
    config["arg"]["video"] = video
    for key, subconfigs in myconfig.items():
        for subkey, subconfig in subconfigs.items():
            config[key][subkey] = subconfig
    dic = {
        "name": name, 
        "folder": create_xp_folder(),
        "config": config,
        "walltime": {},
        "time": {},
        "stop_reason": {},
    }
    return dic 


# ### plot functions 

# +
def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def compute_test(data, coef=1, N=5):
    pairs = []
    for i in range(N-1):
        for j in range(i+1,N):
            pairs.append((i,j))
    res = [[] for _ in range(N)]
    cells = [[ '' for _ in range(N-1)] for _ in range(N-1)]
    for (param1,param2) in pairs:
        data1 = np.array(data[param1])
        data2 = np.array(data[param2])
        stat, p_t = ttest_ind(data1,data2)
        p_t = p_t*coef
        mean1 = np.median(data1)
        mean2 = np.median(data2)
        d = truncate(mean1/mean2, 2)
        if d=='0.00' or d=='-0.00':
            d = truncate( mean1/mean2, 3)
            if d=='0.000' or d=='-0.000':
                d = truncate( mean1/mean2, 4)
                
        if p_t < 0.001:
            res.append((param1, param2, '***'))
            cells[param1][N-1-param2] = d+'\n***'
        elif p_t < 0.01:
            res.append((param1, param2, '**'))
            cells[param1][N-1-param2] = d+'\n**'
        elif p_t < 0.05:
            res.append((param1, param2, '*'))
            cells[param1][N-1-param2] = d+'\n*'
        else:
            cells[param1][N-1-param2] = "ns"
    return res,cells

def plot_boxplot(data, names, ylabel="performance", ylim=None, title="", log=False, bbox=(1.13,0.1,0.5,0.9), 
                 correction=True, rotation=0, use_table=True, ):
    N = len(data)
    stat,cells = compute_test(data, coef=3*N*(N-1)/2 if correction else 1, N=N)
    
    #plt.subplots(figsize=(16,9))
    
    bplot = sb.boxplot(data=data)

    for i in range(N):
        bplot.artists[i].set_facecolor(colors[i])
    if np.size(data) < 1000:
        sb.swarmplot(data=data, color='black', edgecolor='black',size=7)

    plt.grid(axis='y')
    rows = names[:N-1]
    columns = [names[i] for i in range(N-1,0,-1)]
    cell_text = cells
    cellColours = [['white' if N-1-i>j else 'lightgrey' for j in range(N-1)] for i in range(N-1) ]
    if use_table:
        the_table = plt.table(cellText=cell_text,
                              rowLabels=rows,
                              cellColours= cellColours,
                              rowColours=colors[:N-1],
                              colColours=[ colors[i] for i in range(N-1,0,-1)],
                              colLabels=columns,
                              cellLoc = 'center',
                              bbox=bbox)
    
    plt.xticks(range(N), names, rotation=rotation)
    if log:
        plt.yscale('log')
    if not ylim is None:
        plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.title(title)


# -

if os.uname()[1] == "multivac":
    n_proc = 60
elif os.uname()[1] == "evo256":
    n_proc = 250
elif os.uname()[1] == "haretis-42":
    n_proc = 5
else:
    n_proc = 1


# +
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


# -

def pretty_print_condition(condition):
    txt = ""
    for c in condition:
        txt += str(c)
        txt += "&"
    return txt[:-1]


# ## NN

# +
class NN(nn.Module):

    def __init__(self, config):
        super(NN, self).__init__()
        layers = []
        layers_dim = [config["input_dim"]] + config["layers"]
        for i in range(1,len(layers_dim)):
            layers.append(nn.Linear(layers_dim[i-1], layers_dim[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config["dropout"]))
        layers.append(nn.Linear(layers_dim[-1], 1))
        if type(config["criterion"]) != torch.nn.BCEWithLogitsLoss:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.net(x)
        return x
    
def compute_proba_pred(X, model, dropout=False, use_BCE=True):
    model.eval()
    y_pred_prob = 0
    with torch.no_grad():
        y_pred = model.forward(X)
        if use_BCE:
            y_pred_prob = torch.sigmoid(y_pred).cpu()
        else:
            y_pred_prob = y_pred.cpu()
    return y_pred_prob

def predict_map(dic, standardization, model, use_BCE, use_CUDA):
    samples = []
    X, Z = dic['X'], dic['Z']
    for x in X:
        for z in Z:
            samples.append(np.concatenate((np.copy(dic["input"]), [float(x)], [float(z)])))
    samples = torch.tensor(samples, dtype=torch.float)
    samples = (samples - standardization['mean']) / standardization['std']
    if use_CUDA:
        samples = samples.cuda()
    else:
        samples = torch.tensor(np.array(samples))
    proba = compute_proba_pred(samples, model=model, use_BCE=use_BCE)
    k = 0
    proba_map = np.empty((len(Z), len(X)))
    for i, x in enumerate(X):
        for j, z in enumerate(Z):
            proba_map[j][i] = proba[k]
            k+=1
    dic["predict_map"] = proba_map
    z_index, x_index = np.unravel_index(np.argmax(proba_map), proba_map.shape)
    dic["(z,x)"] = z_index, x_index
    
def create_input_(input_name_list, sample):
    input_val_list = []
    for input_name in input_name_list:
        if "__" in input_name:
            l = input_name.split("__")
            if len(l) == 2:
                if l[0] == "uni":
                    input_val_list.append(np.random.uniform(-1,1,int(l[1])))
                elif l[0] == "bool":
                    input_val_list.append(np.random.randint(0,2,int(l[1])))
            else:
                print("bad argument ", input_name)
        else:
            input_val = sample[input_name]
            if type(input_val) == float:
                input_val_list.append([input_val])
            else:
                input_val_list.append(input_val)
    return np.concatenate(input_val_list)


# -

# # Avoided among robust

# ## Load saved model

datapath = "/home/pal/notebooks/data/wall_reflex/"
snapshots_save_folder = "/home/pal/notebooks/data/wall_reflex/snapshots"

with open("/home/pal/notebooks/data/wall_reflex/datasets/ALP_v3_dataset.pk", 'rb') as f:
    dataset = pickle.load(f)

# +
path = "NN_tuning/2022/04/24/10h17m22_7955"
xp = "damage_ALP_v3_epochs_200_input_wall_distance_wall_angle_q"
i = 0
ep = "final"
full_path = f"{datapath+path}/{xp}/{xp}_replicate_{i}"

with open(f"{full_path}/config_ep{ep}.pk", 'rb') as f:
    NN_config = pickle.load(f)

with open(f"{full_path}/evaluation.pk", 'rb') as f:
    evaluation = pickle.load(f)

with open(f"{full_path}/standardization.pk", 'rb') as f:
    standardization = pickle.load(f)
    
with open(f"{full_path}/measures_epfinal.pk", 'rb') as f:
    measures = pickle.load(f)
    
model = NN(NN_config)
model.load_state_dict(torch.load(full_path+"/model_epfinal.trch"))  
use_CUDA = False
use_BCE = NN_config['criterion'] == "BCEWithLogitsLoss"
if use_CUDA:
    model = model.cuda()
# -

x = np.argmax(measures["usage"]["val"])

# +
blue = '#4b4dce'
green = '#117733'
import matplotlib as mpl
font = {'size'   : 30}
mpl.rc('font', **font)

plt.subplots(figsize=(16,9))
plt.plot(np.array(measures["usage"]["train"])*100, lw=10, color=green, label="Training")
plt.plot(np.array(measures["usage"]["val"])*100, lw=10,  color=blue,label="Validation")
m = 100*min(np.min(measures["usage"]["val"]), np.min(measures["usage"]["train"]))
M = 100*max(np.max(measures["usage"]["val"]), np.max(measures["usage"]["train"]))
plt.vlines(x=x, ymin=0, ymax=100, lw=3, color=blue, ls=":", label="Best Validation")
plt.legend()
plt.grid(axis='y')
plt.ylim((0,100))
plt.ylabel("Success rate (%)")
plt.xlabel("Epochs")
    #plt.savefig("/home/tim/Sync/Figures/2022/02/23/example_NN_training.png")
# -

params = {}
for key in evaluation["NN_res"]["avoided among robust"]:
    params[key] = dataset[key]

for dic in tqdm(params.values()):
    dic['input'] = create_input_(NN_config['input'], dic)
    predict_map(dic, standardization, model, use_BCE, use_CUDA)

# ## Select Xp

for xp in tqdm(range(min(100,len(params)))):
    key = list(params.keys())[xp]
    dic = params[key]
    condition = dic["condition"]
    rh, lh = [0, 0, 0], [0, 0, 0]
    (rh[0],rh[1],rh[2],lh[0],lh[1],lh[2],wall_angle, wall_distance) = key 

    X = dic['X']
    Z = dic['Z']
    z_id, x_id = dic['(z,x)']

    data_dicts = {}

    name = f"test"
    x0 =  float(wall_distance*np.sin(wall_angle))
    y0 =  -float(wall_distance*np.cos(wall_angle))
    x = X[x_id]
    z = Z[z_id]
    posx = x0 + float(x*np.cos(wall_angle))
    posy = y0 + float(x*np.sin(wall_angle))
    dist = float(np.sqrt(posx**2+posy**2))
    coef_reducteur = 1-0.1/dist  # pour mettre la tâche de la boule environ 10cm avant le mur 
    posx = coef_reducteur * posx
    posy = coef_reducteur * posy 

    myconfig = {
        "CONTROLLER": {  
            "duration": 5,
            "colision_shapes": [[False, [x0, y0, 1.1, 0., 0., wall_angle, 5, 0.05, 2.]]],
            "damage_time": 4.,
            "reflex_time": 4.002,
            "use_baseline": False,
            "log_level": 0, 
            "reflex_x": float(x),
            "wall_distance": wall_distance,
            "wall_angle": wall_angle, 
            "use_reflex_trajectory": False,
            "update_contact_when_detected": True,
            "remove_rf_tasks": False,
            "reflex_arm_stiffness": 1.,
            "reflex_stiffness_alpha": 0.9999,
            "wall_distance": wall_distance,
            "wall_angle": wall_angle, 
            "condition": condition,
            "use_left_hand": False, 
            "use_right_hand": True,  
            "urdf": "talos_fast_collision(RAL_revision).urdf",
            "cam_pos": [3.5, 2, 2.2, 0., 0., 1.3],
            "dt": 0.002,
        }, 
        "TASK": {
            "contact_rhand": {
                "x": float(posx),
                "y": float(posy),
                "z": float(z),
                "kp": 30., 
            },
            "lh": {
                "weight": 1000.,
            },
            "rh": {
                "weight": 1000.,
            },
        },
        "BEHAVIOR": {
            "lh_shift": [float(lh[0]), float(lh[1]), float(lh[2]), 0., 0., 0.],
            "rh_shift": [float(rh[0]), float(rh[1]), float(rh[2]), 0., 0., 0.],
            "trajectory_duration": 4.,
        },
        "arg": {
            "recording": True,
            "video": False,

        },
    }
    data_dicts[name] = test_default(name, actuator="spd", myconfig=myconfig)
    data_dicts[name]["conditions"] = [[Conditions["A"][c] for c in condition]]
    jobs, n_jobs = make_jobs_custom_conditions(data_dicts, verbose=0)

    q_folder = master(data_dicts, jobs, n_jobs, n_processes=n_proc, verbose=0)
    damage = os.listdir(q_folder+"/"+name)
    video_folder = q_folder+"/"+name+"/"+damage[0]
    video_path = video_folder+"/video.mp4"

    save_folder = video_folder +"/snapshots"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    cam = cv2.VideoCapture(video_path)
    ret, frame = cam.read()
    i = 0
    while ret:
        previous_frame = frame
        if i % 10 == 0:
            name = f"{save_folder}/frame_{i}.png"    
            cv2.imwrite(name, previous_frame)
        i+=1
        ret, frame = cam.read()

    images_folder = save_folder
    ims = []
    x, y = 0.22, 0.22

    pad_x, pad_y = 0., 0.1
    #os.listdir(images_folder)
    if os.path.exists(f"{images_folder}/frame_180.png"):
        for image in ["frame_160.png", "frame_180.png"]:
            if ".png" in image:
                image = Image.open(f"{images_folder}/{image}")
                bbox = image.getbbox()
                lx, ly = bbox[2]*x, bbox[2]*y
                x_center, y_center = bbox[2]*(0.5+pad_x), bbox[3]*(0.5+pad_y)
                cropped = (x_center-lx, y_center-ly, x_center+lx, y_center+ly)
                ims.append(image.crop(cropped))

        h = ims[0]
        for i in range(1, len(ims)):
            h = get_concat_h(h, ims[i])

            d = 9.81
        fig, ax = plt.subplots(figsize=(d,d))
        ax.pcolor(dic["map"], cmap="viridis", edgecolors='k', vmin=0, vmax=1, linewidths=1)
        ax.scatter(x_id +0.5, z_id+0.5, marker="x", s=500, color="black")
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.gca().set_axis_off()
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_folder+"/plot2.png", pad_inches = 0)
        plt.close()
        fig, ax = plt.subplots(figsize=(d,d))
        ax.pcolor(dic["predict_map"], cmap="viridis", edgecolors='k', vmin=np.min(dic["predict_map"]), vmax=np.max(dic["predict_map"]), linewidths=1)
        line = ax.scatter(x_id +0.5, z_id+0.5, marker="x", s=500, color="black")
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.gca().set_axis_off()
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_folder+"/plot1.png", pad_inches = 0)
        plt.close()
        plot_image1 = Image.open(save_folder+"/plot1.png")
        l = plot_image1.width*0.4
        x_center, y_center = plot_image1.width*0.5, plot_image1.height*0.5
        cropped = (x_center-l, y_center-l, x_center+l, y_center+l)
        plot_image1 = plot_image1.crop(cropped)
        plot_image2 = Image.open(save_folder+"/plot2.png")
        plot_image2 = plot_image2.crop(cropped)
        final = get_concat_h(h, plot_image1)
        final = get_concat_h(final, plot_image2)

        final.save(snapshots_save_folder+f"/RAL_Revision/success/{xp}_{pretty_print_condition(condition)}.png")

# # Unavoided among robust

params = {}
for key in evaluation["NN_res"]["unavoided among robust"]:
    params[key] = dataset[key]

for dic in tqdm(params.values()):
    dic['input'] = create_input_(NN_config['input'], dic)
    predict_map(dic, standardization, model, use_BCE, use_CUDA)

for xp in tqdm(range(min(100,len(params)))):
    key = list(params.keys())[xp]
    dic = params[key]
    condition = dic["condition"]
    rh, lh = [0, 0, 0], [0, 0, 0]
    (rh[0],rh[1],rh[2],lh[0],lh[1],lh[2],wall_angle, wall_distance) = key 

    X = dic['X']
    Z = dic['Z']
    z_id, x_id = dic['(z,x)']

    data_dicts = {}

    name = f"test"
    x0 =  float(wall_distance*np.sin(wall_angle))
    y0 =  -float(wall_distance*np.cos(wall_angle))
    x = X[x_id]
    z = Z[z_id]
    posx = x0 + float(x*np.cos(wall_angle))
    posy = y0 + float(x*np.sin(wall_angle))
    dist = float(np.sqrt(posx**2+posy**2))
    coef_reducteur = 1-0.1/dist  # pour mettre la tâche de la boule environ 10cm avant le mur 
    posx = coef_reducteur * posx
    posy = coef_reducteur * posy 

    myconfig = {
        "CONTROLLER": {  
            "duration": 15,
            "colision_shapes": [[False, [x0, y0, 1.1, 0., 0., wall_angle, 5, 0.05, 2.]]],
            "damage_time": 4.,
            "reflex_time": 4.002,
            "use_baseline": False,
            "log_level": 0, 
            "reflex_x": float(x),
            "wall_distance": wall_distance,
            "wall_angle": wall_angle, 
            "use_reflex_trajectory": False,
            "update_contact_when_detected": True,
            "remove_rf_tasks": False,
            "reflex_arm_stiffness": 1.,
            "reflex_stiffness_alpha": 0.9999,
            "wall_distance": wall_distance,
            "wall_angle": wall_angle, 
            "condition": condition,
            "use_left_hand": False, 
            "use_right_hand": True,  
            "urdf": "talos_fast_collision(RAL_revision).urdf",
            "cam_pos": [3.5, 2, 2.2, 0., 0., 1.3],
            "dt": 0.002,
        }, 
        "TASK": {
            "contact_rhand": {
                "x": float(posx),
                "y": float(posy),
                "z": float(z),
                "kp": 30., 
            },
            "lh": {
                "weight": 1000.,
            },
            "rh": {
                "weight": 1000.,
            },
        },
        "BEHAVIOR": {
            "lh_shift": [float(lh[0]), float(lh[1]), float(lh[2]), 0., 0., 0.],
            "rh_shift": [float(rh[0]), float(rh[1]), float(rh[2]), 0., 0., 0.],
            "trajectory_duration": 4.,
        },
        "arg": {
            "recording": True,
            "video": False,

        },
    }
    data_dicts[name] = test_default(name, actuator="spd", myconfig=myconfig)
    data_dicts[name]["conditions"] = [[Conditions["A"][c] for c in condition]]
    jobs, n_jobs = make_jobs_custom_conditions(data_dicts, verbose=0)

    q_folder = master(data_dicts, jobs, n_jobs, n_processes=n_proc, verbose=0)
    damage = os.listdir(q_folder+"/"+name)
    video_folder = q_folder+"/"+name+"/"+damage[0]
    video_path = video_folder+"/video.mp4"

    save_folder = video_folder +"/snapshots"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    cam = cv2.VideoCapture(video_path)
    ret, frame = cam.read()
    i = 0
    last_frame = 0
    while ret:
        previous_frame = frame
        if i % 10 == 0:
            name = f"{save_folder}/frame_{i}.png"    
            cv2.imwrite(name, previous_frame)
            last_frame = f"frame_{i}.png"
        i+=1
        ret, frame = cam.read()

    images_folder = save_folder
    ims = []
    x, y = 0.22, 0.22

    pad_x, pad_y = 0., 0.1
    #os.listdir(images_folder)
    if os.path.exists(f"{images_folder}/{last_frame}") and os.path.exists(f"{images_folder}/frame_160.png"):
        for image in ["frame_160.png", last_frame]:
            if ".png" in image:
                image = Image.open(f"{images_folder}/{image}")
                bbox = image.getbbox()
                lx, ly = bbox[2]*x, bbox[2]*y
                x_center, y_center = bbox[2]*(0.5+pad_x), bbox[3]*(0.5+pad_y)
                cropped = (x_center-lx, y_center-ly, x_center+lx, y_center+ly)
                ims.append(image.crop(cropped))

        h = ims[0]
        for i in range(1, len(ims)):
            h = get_concat_h(h, ims[i])

            d = 9.81
        fig, ax = plt.subplots(figsize=(d,d))
        ax.pcolor(dic["map"], cmap="viridis", edgecolors='k', vmin=0, vmax=1, linewidths=1)
        ax.scatter(x_id +0.5, z_id+0.5, marker="x", s=500, color="black")
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.gca().set_axis_off()
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_folder+"/plot2.png", pad_inches = 0)
        plt.close()
        fig, ax = plt.subplots(figsize=(d,d))
        ax.pcolor(dic["predict_map"], cmap="viridis", edgecolors='k', vmin=np.min(dic["predict_map"]), vmax=np.max(dic["predict_map"]), linewidths=1)
        line = ax.scatter(x_id +0.5, z_id+0.5, marker="x", s=500, color="black")
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.gca().set_axis_off()
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_folder+"/plot1.png", pad_inches = 0)
        plt.close()
        plot_image1 = Image.open(save_folder+"/plot1.png")
        l = plot_image1.width*0.4
        x_center, y_center = plot_image1.width*0.5, plot_image1.height*0.5
        cropped = (x_center-l, y_center-l, x_center+l, y_center+l)
        plot_image1 = plot_image1.crop(cropped)
        plot_image2 = Image.open(save_folder+"/plot2.png")
        plot_image2 = plot_image2.crop(cropped)
        final = get_concat_h(h, plot_image1)
        final = get_concat_h(final, plot_image2)

        final.save(snapshots_save_folder+f"/RAL_Revision/unavoided_among_robust/{xp}_{pretty_print_condition(condition)}.png")

# # Unavoidable

params = {}
for key, dic in dataset.items():
    if np.sum(dic['map']) == 0: 
        params[key] = dataset[key]

counter = 0
for xp in tqdm(range(min(100,len(params)))):
    key, dic = list(params.items())[xp]
    condition = dic["condition"]
    rh, lh = [0, 0, 0], [0, 0, 0]
    (rh[0],rh[1],rh[2],lh[0],lh[1],lh[2],wall_angle, wall_distance) = key 
    x0 = float(wall_distance*np.sin(wall_angle))
    y0 = -float(wall_distance*np.cos(wall_angle))
    data_dicts = {}
    name = f"test"
    myconfig = {
        "CONTROLLER": {  
            "duration": 4.004,
            "colision_shapes": [[False, [x0, y0, 1.1, 0., 0., wall_angle, 5, 0.05, 2.]]],
            "damage_time": 4.,
            "reflex_time": 4.002,
            "use_baseline": False,
            "log_level": 0, 
            "reflex_x": float(x),
            "use_reflex_trajectory": False,
            "update_contact_when_detected": True,
            "remove_rf_tasks": False,
            "reflex_arm_stiffness": 1.,
            "reflex_stiffness_alpha": 0.9999,
            "wall_distance": wall_distance,
            "wall_angle": wall_angle, 
            "condition": condition,
            "use_left_hand": False, 
            "use_right_hand": True,  
            "urdf": "talos_fast_collision(RAL_revision).urdf",
            "cam_pos": [3.5, 2, 2.2, 0., 0., 1.3],
            "dt": 0.002,
        }, 
        "TASK": {
            "lh": {
                "weight": 1000.,
            },
            "rh": {
                "weight": 1000.,
            },
        },
        "BEHAVIOR": {
            "lh_shift": [float(lh[0]), float(lh[1]), float(lh[2]), 0., 0., 0.],
            "rh_shift": [float(rh[0]), float(rh[1]), float(rh[2]), 0., 0., 0.],
            "trajectory_duration": 4.,
        },
        "arg": {
            "recording": True,
            "video": False,

        },
    }
    data_dicts[name] = test_default(name, actuator="spd", myconfig=myconfig)
    data_dicts[name]["conditions"] = [[Conditions["A"][c] for c in condition]]
    jobs, n_jobs = make_jobs_custom_conditions(data_dicts, verbose=0)

    q_folder = master(data_dicts, jobs, n_jobs, n_processes=n_proc, verbose=0)

    damage = os.listdir(q_folder+"/"+name)
    video_folder = q_folder+"/"+name+"/"+damage[0]
    video_path = video_folder+"/video.mp4"

    save_folder = video_folder +"/snapshots"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    cam = cv2.VideoCapture(video_path)
    ret, frame = cam.read()
    while ret:
        previous_frame = frame  
        ret, frame = cam.read()
    name = f"{save_folder}/frame_{i}.png"    
    cv2.imwrite(name, previous_frame)
    x, y = 0.22, 0.22
    pad_x, pad_y = 0., 0.1
    image = Image.open(name)
    bbox = image.getbbox()
    lx, ly = bbox[2]*x, bbox[2]*y
    x_center, y_center = bbox[2]*(0.5+pad_x), bbox[3]*(0.5+pad_y)
    cropped = (x_center-lx, y_center-ly, x_center+lx, y_center+ly)
    h = image.crop(cropped)
    h.save(snapshots_save_folder+f"/RAL_Revision/unavoidable/{xp}_{pretty_print_condition(condition)}.png")

# # posture without cross

params = {}
for key in evaluation["NN_res"]["unavoided among robust"]:
    params[key] = dataset[key]

for dic in tqdm(params.values()):
    dic['input'] = create_input_(NN_config['input'], dic)
    predict_map(dic, standardization, model, use_BCE, use_CUDA)

for xp in tqdm(range(min(40,len(params)))):
    key = list(params.keys())[xp]
    dic = params[key]
    condition = dic["condition"]
    rh, lh = [0, 0, 0], [0, 0, 0]
    (rh[0],rh[1],rh[2],lh[0],lh[1],lh[2],wall_angle, wall_distance) = key 

    X = dic['X']
    Z = dic['Z']
    z_id, x_id = dic['(z,x)']

    data_dicts = {}

    name = f"test"
    x0 =  float(wall_distance*np.sin(wall_angle))
    y0 =  -float(wall_distance*np.cos(wall_angle))
    x = X[x_id]
    z = Z[z_id]
    posx = x0 + float(x*np.cos(wall_angle))
    posy = y0 + float(x*np.sin(wall_angle))
    dist = float(np.sqrt(posx**2+posy**2))
    coef_reducteur = 1-0.1/dist  # pour mettre la tâche de la boule environ 10cm avant le mur 
    posx = coef_reducteur * posx
    posy = coef_reducteur * posy 

    myconfig = {
        "CONTROLLER": {  
            "duration": 5.004,
            "colision_shapes": [[False, [x0, y0, 1.1, 0., 0., wall_angle, 5, 0.05, 2.]]],
            "damage_time": 4.,
            "reflex_time": 4.002,
            "use_baseline": False,
            "log_level": 0, 
            "reflex_x": float(x),
            "wall_distance": wall_distance,
            "wall_angle": wall_angle, 
            "use_reflex_trajectory": False,
            "update_contact_when_detected": True,
            "remove_rf_tasks": False,
            "reflex_arm_stiffness": 1.,
            "reflex_stiffness_alpha": 0.9999,
            "wall_distance": wall_distance,
            "wall_angle": wall_angle, 
            "condition": condition,
            "use_left_hand": False, 
            "use_right_hand": True,  
            "urdf": "talos_fast_collision(RAL_revision).urdf",
            "cam_pos": [3.5, 2, 2.2, 0., 0., 1.3],
            "dt": 0.002,
        }, 
        "TASK": {
            "contact_rhand": {
                "x": float(posx),
                "y": float(posy),
                "z": float(z),
                "kp": 30., 
            },
            "lh": {
                "weight": 1000.,
            },
            "rh": {
                "weight": 1000.,
            },
        },
        "BEHAVIOR": {
            "lh_shift": [float(lh[0]), float(lh[1]), float(lh[2]), 0., 0., 0.],
            "rh_shift": [float(rh[0]), float(rh[1]), float(rh[2]), 0., 0., 0.],
            "trajectory_duration": 4.,
        },
        "arg": {
            "recording": True,
            "video": False,

        },
    }
    data_dicts[name] = test_default(name, actuator="spd", myconfig=myconfig)
    data_dicts[name]["conditions"] = [[Conditions["A"][c] for c in condition]]
    jobs, n_jobs = make_jobs_custom_conditions(data_dicts, verbose=0)

    q_folder = master(data_dicts, jobs, n_jobs, n_processes=n_proc, verbose=0)
    damage = os.listdir(q_folder+"/"+name)
    video_folder = q_folder+"/"+name+"/"+damage[0]
    video_path = video_folder+"/video.mp4"

    save_folder = video_folder +"/snapshots"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    cam = cv2.VideoCapture(video_path)
    ret, frame = cam.read()
    i = 0
    last_frame = 0
    while ret:
        previous_frame = frame
        if i % 10 == 0:
            name = f"{save_folder}/frame_{i}.png"    
            cv2.imwrite(name, previous_frame)
            last_frame = f"frame_{i}.png"
        i+=1
        ret, frame = cam.read()

    images_folder = save_folder
    ims = []
    x, y = 0.22, 0.22

    pad_x, pad_y = 0., 0.1
    #os.listdir(images_folder)
    if os.path.exists(f"{images_folder}/frame_160.png"):
        for image in ["frame_160.png"]:
            if ".png" in image:
                image = Image.open(f"{images_folder}/{image}")
                bbox = image.getbbox()
                lx, ly = bbox[2]*x, bbox[2]*y
                x_center, y_center = bbox[2]*(0.5+pad_x), bbox[3]*(0.5+pad_y)
                cropped = (x_center-lx, y_center-ly, x_center+lx, y_center+ly)
                ims.append(image.crop(cropped))

        h = ims[0]
        d = 9.81
        fig, ax = plt.subplots(figsize=(d,d))
        ax.pcolor(dic["map"], cmap="viridis", edgecolors='k', vmin=0, vmax=1, linewidths=1)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.gca().set_axis_off()
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_folder+"/plot2.png", pad_inches = 0)
        plt.close()
        
        plot_image2 = Image.open(save_folder+"/plot2.png")
        h.save(snapshots_save_folder+f"/RAL_Revision/no_cross/{xp}_{pretty_print_condition(condition)}_posture.png")
        plot_image2.save(snapshots_save_folder+f"/RAL_Revision/no_cross/{xp}_{pretty_print_condition(condition)}_truth.png")        


