# -*- coding: utf-8 -*-
# # Config

import numpy as np 
import pickle 
import os 
import torch
import sys
from tqdm import tqdm
sys.path.append("/home/haretis/Experience/DeepLearning/TCN/")
from torch import nn
from torch.autograd import Variable
from time import sleep, time 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler   
from copy import deepcopy
import argparse
from datetime import datetime
import scipy.signal
from sklearn.decomposition import PCA 
import seaborn as sb
from numpy.random import default_rng 
import matplotlib as mpl
rng = default_rng()


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


if os.uname()[1] == "multivac":
    datapath = "/home/tim/Experiences/Deeplearning/notebooks/data/wall_reflex/" 
elif os.uname()[1] == "haretis-42":
    datapath = "/home/haretis/Experiences/Deeplearning/notebooks/data/wall_reflex/"
else:
    # G5k 
    datapath = "/home/tanne/Experiences/notebooks/data/wall_reflex/"

# +
font = {'size'   : 18}
mpl.rc('font', **font)

plt.rcParams["figure.figsize"] = (16, 9)


# -

# # NN 

# ## Loading data

# ### load_data

# +
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


def load_data(config, idx=0):
    with open(datapath+"datasets/"+config["damage"]+"_dataset.pk", "rb") as f:
        data = pickle.load(f)
    In, Out = [], []
    condition_maps = {}
    for i in tqdm(rng.permutation(len(data))[:config["n_samples"]]):
        sample = list(data.values())[i]
        key = list(data.keys())[i]
        input_data = create_input_(config['input'], sample)
        condition_map = {}
        T = sample["map"]
        X = sample["X"]
        Z = sample["Z"] 
        for iz, z in enumerate(sample["Z"]):
            for ix, x in enumerate(sample["X"]):
                In.append(np.concatenate((np.copy(input_data), [x], [z])))
                Out.append([T[iz,ix]])
                condition_map[(x, z)] = idx
                idx += 1
        condition_maps[i] = {"key": key, 
                             "indices" : condition_map, 
                             "map": T, 
                             "input": input_data, 
                             "truth": sample["truth"], 
                             "X": X, 
                             "Z": Z}
    return In, Out, condition_maps


# -

# ### data generator

def data_generator(config):
    X, Y, condition_maps = load_data(config)
    X = torch.tensor(X, dtype=torch.float)
    if config["standardize"]:
        means = X.mean(dim=(0), keepdim=True)
        stds = X.std(dim=(0), keepdim=True) + 1e-10
        X = (X - means) / stds
        standardization = {"mean": means, "std": stds}
        with open(config["logdir"]+f"/standardization.pk", 'wb') as f:
            pickle.dump(standardization, f)
    else:
        standardization = None

    return Variable(X), Variable(torch.tensor(Y, dtype=torch.float)), condition_maps, standardization


# ### generate_train_test

def generate_train_test(config, permutation=None, training=True):
    X, Y, condition_maps, standardization = data_generator(config)
    if permutation is None:
        permutation = rng.permutation(len(condition_maps))
    if (config["test_ratio"]+config["val_ratio"]>=1):
        print("Empty training set")
    n_test = int(len(permutation)*config["test_ratio"])
    n_val = int(len(permutation)*config["val_ratio"])
    condition_idx_train, condition_idx_val, condition_idx_test = permutation[n_test+n_val:], permutation[n_test:n_test+n_val], permutation[:n_test]
    list_conditions = list(condition_maps.values())
    sample_idx_train, sample_idx_test, sample_idx_val = [], [], []
    maps_train, maps_test, maps_val = [], [], []
    for i in condition_idx_train:
        sample_idx_train = np.concatenate((sample_idx_train,list(list_conditions[i]["indices"].values())))
        maps_train.append(list_conditions[i])
    for i in condition_idx_test:
        sample_idx_test = np.concatenate((sample_idx_test,list(list_conditions[i]["indices"].values())))
        maps_test.append(list_conditions[i])
    for i in condition_idx_val:
        sample_idx_val = np.concatenate((sample_idx_val,list(list_conditions[i]["indices"].values())))
        maps_val.append(list_conditions[i])
        
    X_train, Y_train = X[sample_idx_train], Y[sample_idx_train]
    X_test, Y_test = X[sample_idx_test], Y[sample_idx_test]
    X_val, Y_val = X[sample_idx_val], Y[sample_idx_val]
    config["input_channels"] = X_train.shape[1] 
    if config["cuda"]:
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()
        X_val = X_val.cuda()
        Y_val = Y_val.cuda()
        X_test = X_test.cuda()
        Y_test = Y_test.cuda()
    X = {"train": X_train, "val": X_val, "test": X_test}
    Y = {"train": Y_train, "val": Y_val, "test": Y_test}
    maps = {"train": maps_train, "val": maps_val, "test": maps_test}
    return X, Y, permutation, standardization, maps


# ## Model 

# ### Model NN

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


# ### generate model

# +
class WrongCriterionError(Exception):
    pass 

def generate_model(config, Y_train):
    model = NN(config)
    if config["cuda"]:
        model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    if config["criterion"] == "BCEWithLogitsLoss":
        pos_weight = (len(Y_train)-torch.sum(Y_train).cpu())/torch.sum(Y_train).cpu()  # virtually balance the positive and negative examples 
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif config["criterion"] == "MSELoss":
        criterion = torch.nn.nn.MSELoss()
    elif config["criterion"] == "WeightedMSELoss":
        def compute_weighted_mse_loss(Y_train):
            pos_weight = (len(Y_train)-torch.sum(Y_train).cpu())/torch.sum(Y_train).cpu()
            def weighted_mse_loss(input, target): 
                return ((1+(pos_weight-1)*target) * (input - target) ** 2).mean()
            return weighted_mse_loss
        criterion = compute_weighted_mse_loss(Y_train)
    else:
        raise WrongCriterionError
    return model, optimizer, criterion


# -

# ## Training functions

# ###  Classification evaluation

def binary_acc(y_pred, y_test, use_BCE=True):
    if use_BCE:
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
    else:
        y_pred_tag = torch.round(y_pred)
    y_test = torch.round(y_test)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    return acc


def confusion_matrix(y_pred, y_test, use_BCE=True):
    if use_BCE:
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
    else:
        y_pred_tag = torch.round(y_pred)
    y_test = torch.round(y_test)
    cfm = np.zeros((2,2))
    for i, y in enumerate(y_test):
        cfm[int(y_pred_tag[i])][int(y)] += 1 
    return cfm


def MCC(cfm):
    """ Matthews correlation coefficient """
    TP = cfm[1,1]
    TN = cfm[0,0]
    FP = cfm[1,0]
    FN = cfm[0,1]
    denominator = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    if denominator:
        mcc = (TP*TN-FP*FN)/denominator
    else:
        mcc = 0.
    return mcc


# ### evaluate

def evaluate(model, X, Y, use_BCE=True):
    model.eval()
    with torch.no_grad():
        output = model(X)
        loss = criterion(output, Y)
        acc = binary_acc(output, Y, use_BCE)
        mcc = MCC(confusion_matrix(output, Y, use_BCE))
        return loss.cpu(), acc.cpu(), mcc 


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


def prepare_test_set(raw_test_set):
    Samples = []
    for dic in tqdm(raw_test_set):
        T = dic["map"]
        min_pooled = np.zeros(T.shape)
        for z in range(len(T)):
            for x in range(len(T[z])):
                min_pooled[z,x] = np.min(T[max(0, z-1):z+2, max(0, x-1):x+2])
        if np.sum(min_pooled) > 0:
            samples = []
            input_data = dic["input"]
            X = dic["X"]
            Z = dic["Z"]
            for x in X:
                for z in Z: 
                    samples.append(np.concatenate((np.copy(input_data), [float(x)], [float(z)])))
            samples = torch.tensor(samples, dtype=torch.float)
            # Standardize
            if standardization is not None:
                samples = (samples - standardization['mean']) / standardization['std']
            # Infer 
            samples = samples.cuda()
            Samples.append((samples, T, X, Z))
    return Samples 


def evaluate_usage(model, test_set, use_BCE=True):
    score = 0
    for [samples, truth, X, Z] in test_set:
        proba = compute_proba_pred(samples, model=model, use_BCE=use_BCE)
        k = 0
        proba_map = np.empty((len(Z), len(X)))
        for i, x in enumerate(X):
            for j, z in enumerate(Z):
                proba_map[j][i] = proba[k]
                k+=1
        z_index, x_index = np.unravel_index(np.argmax(proba_map), proba_map.shape)
        score += int(truth[z_index, x_index])
    return score/len(test_set)


# ### train

def train(epoch, model, optimizer, criterion, X_train, Y_train, verbose=0):
    model.train()
    batch_idx = 1
    batch_size = config["batch_size"]
    for i in range(0, X_train.size(0), batch_size):
        if i + batch_size > X_train.size(0):
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i+batch_size)], Y_train[i:(i+batch_size)]
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        if config['clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
        optimizer.step()


# ### learning

def learning(dic, X_train, Y_train, X_val, Y_val, X_test, Y_test, maps):
    config, model, optimizer, criterion = dic['config'], dic["model"], dic["optimizer"], dic["criterion"]
    loss = {"train": [], "val": [], "test": []}
    acc = {"train": [], "val": [], "test": []}
    mcc = {"train": [], "val": [], "test": []}
    usage = {"train": [], "val": [], "test": []}
    use_BCE = type(config["criterion"]) == torch.nn.BCEWithLogitsLoss
    tmp_model = deepcopy(model)
    best_loss = {"value": np.inf, "epoch": 0., "model": tmp_model.cpu().state_dict()}
    best_acc = {"value": 0., "epoch": 0., "model": tmp_model.cpu().state_dict()}
    best_mcc = {"value": -1., "epoch": 0., "model": tmp_model.cpu().state_dict()}
    best_usage = {"value": 0., "epoch": 0., "model": tmp_model.cpu().state_dict()}
    train_set = prepare_test_set(maps["train"])
    val_set = prepare_test_set(maps["val"])
    t = tqdm(range(config["epochs"]), ncols=150)
    for ep in t:
        train(ep, model, optimizer, criterion, X_train, Y_train)
        train_loss, train_acc, train_mcc = evaluate(model, X_train, Y_train, use_BCE)
        val_loss, val_acc, val_mcc = evaluate(model, X_val, Y_val, use_BCE)
        train_usage = evaluate_usage(model, train_set, use_BCE=use_BCE)
        val_usage = evaluate_usage(model, val_set, use_BCE=use_BCE)
        if val_loss < best_loss["value"]:
            best_loss["value"] = val_loss
            best_loss["epoch"] = ep
            tmp_model = deepcopy(model)
            best_loss["model"] = deepcopy(tmp_model.cpu().state_dict())
        if val_acc > best_acc["value"]:
            best_acc["value"] = val_acc
            best_acc["epoch"] = ep
            tmp_model = deepcopy(model)
            best_acc["model"] = deepcopy(tmp_model.cpu().state_dict())
        if val_mcc > best_mcc["value"]:
            best_mcc["value"] = val_mcc
            best_mcc["epoch"] = ep
            tmp_model = deepcopy(model)
            best_mcc["model"] = deepcopy(tmp_model.cpu().state_dict())
        if val_usage > best_usage["value"]:
            best_usage["value"] = val_usage
            best_usage["epoch"] = ep
            tmp_model = deepcopy(model)
            best_usage["model"] = deepcopy(tmp_model.cpu().state_dict())
        loss["train"].append(train_loss)
        loss["val"].append(val_loss)
        acc["train"].append(train_acc)
        acc["val"].append(val_acc)
        mcc["train"].append(train_mcc)
        mcc["val"].append(val_mcc)
        usage["train"].append(train_usage)
        usage["val"].append(val_usage)
        t.set_description(f"Trn acc:{train_acc*100:2.1f} mcc:{train_mcc*100:2.1f}% usage:{train_usage*100:2.1f}% Val acc:{val_acc*100:2.1f}%[{best_acc['value']*100:2.1f}%] mcc:{val_mcc*100:2.1f}%[{best_mcc['value']*100:2.1f}%] usage:{val_usage*100:2.1f}%[{best_usage['value']*100:2.1f}%])")
        if ep % config["save_frequency"] == 0 and ep > 0:
            save(config, best_mcc, loss, acc, mcc, usage, dic, ep)
    dic["loss"] = loss
    dic["acc"] = acc
    dic["mcc"] = mcc
    dic["usage"] = usage
    dic["best_loss"] = best_loss
    dic["best_acc"] = best_acc
    dic["best_mcc"] = best_mcc
    dic["best_usage"] = best_usage
    save(config, best_usage, acc, loss, mcc, usage, dic, "final")


# ### save

def save(config, best, loss, acc, mcc, usage, dic, ep):
    with open(f'{config["logdir"]}/measures_ep{ep}.pk', "wb") as f:
        pickle.dump({"acc": acc, "loss":loss, "mcc": mcc, "usage": usage}, f)
    with open(f'{config["logdir"]}/config_ep{ep}.pk', "wb") as f:
        pickle.dump(dic['config'], f)
    torch.save(best["model"], f'{config["logdir"]}/model_ep{ep}.trch')


def clean_save(path, config,  model, standardization):
    with open(f'{path}/config.pk', "wb") as f:
        pickle.dump(config, f)
    with open(f'{path}/standardization.pk', "wb") as f:
        pickle.dump(standardization, f)
    torch.save(model.cpu().state_dict(), f'{path}/model.trch')


# # Config

config = {
    # XP 
    "name": None, 
    "damage": None, 
    "logdir": "",
    "save_frequency": 1000, 
    
    # data 
    "n_classes": 1,
    "standardize": True,
    "n_samples": 2000,
    "input": ["wall_distance", "wall_angle", "q", "dq", "J"],
    
    # Model
    "n_hidden": 1024, # 256 test
    "levels": 2, # 3 test
    
    # Training
    "batch_size": 32,
    "epochs": 2, 
    "dropout": 0.2, # 0. test
    "clip": -1,
    'lr': 1e-5,
    'weight_decay': 0., #0. test
    "criterion": "BCEWithLogitsLoss", #in [BCEWithLogitsLoss, WeightedMSELoss, MSELoss]
    "test_ratio": 0.5,
    "val_ratio": 0.125,
    "cuda": True,
}

# ## Parse args

if config["damage"] is None:
    config["damage"] = "ALP_v4"
if config["name"] is None:
    config["name"] = config["damage"]

if not is_interactive():
    parser = argparse.ArgumentParser()
    # XP
    parser.add_argument("-logdir", type=str)
    parser.add_argument("-name", type=str)
    parser.add_argument("-damage", type=str)
    # Data
    parser.add_argument("-convolution", type=str)
    parser.add_argument("-input", nargs='+', type=str)
    #Model
    parser.add_argument("-n_hidden", type=int)
    parser.add_argument("-levels", type=int)
    # Training
    parser.add_argument("-epochs", type=int)
    parser.add_argument("-dropout", type=float)
    parser.add_argument("-lr", type=float)
    parser.add_argument("-weight_decay", type=float)
    parser.add_argument("-val_ratio", type=float)
    parser.add_argument("-cuda", type=bool)
    parser.add_argument("-criterion", type=str)
    
    arguments = parser.parse_args()
    
    # XP
    if arguments.logdir is not None: config['logdir'] = arguments.logdir
    if arguments.name is not None: config['name'] = arguments.name
    if arguments.damage is not None: config['damage'] = arguments.damage
    # Data
    if arguments.convolution is not None: config['convolution'] = arguments.convolution
    if arguments.input is not None: config['input'] = arguments.input 
    #Model
    if arguments.n_hidden is not None: config['n_hidden'] = arguments.n_hidden
    if arguments.levels is not None: config['levels'] = arguments.levels
    # Training
    if arguments.epochs is not None: config['epochs'] = arguments.epochs
    if arguments.dropout is not None: config['dropout'] = arguments.dropout
    if arguments.lr is not None: config['lr'] = arguments.lr
    if arguments.weight_decay is not None: config['weight_decay'] = arguments.weight_decay
    if arguments.val_ratio is not None: config['val_ratio'] = arguments.val_ratio
    if arguments.cuda is not None: config['cuda'] = arguments.cuda
    if arguments.criterion is not None: config['criterion'] = arguments.criterion
else:
    now = datetime.now()
    timestamp = now.strftime("%Y/%m/%d/%Hh%Mm%Ss")
    logdir = f"{datapath}{timestamp}/{config['name']}/{config['name']}_replicate_{0}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    config["logdir"] = logdir 
    print(logdir)

# ## Check config

config["layers"] = [config["n_hidden"] for _ in range(config["levels"])]
dic = {"config": config}

# # Loading data

X, Y, permutation, standardization, maps = generate_train_test(config)
X_train, Y_train = X["train"], Y["train"]
X_val, Y_val = X["val"], Y["val"]
X_test, Y_test = X["test"], Y["test"]

print(f"training success: {Y_train.sum()/len(Y_train)*100:2.1f}%")
print(f"validation success: {Y_val.sum()/len(Y_val)*100:2.1f}%")

config["input_dim"] = X_train.shape[1] 
print(config["input_dim"] )

# # Actual Training

model, optimizer, criterion = generate_model(dic['config'], Y_train)
dic["model"] = model
dic["optimizer"] = optimizer
dic["criterion"] = criterion

learning(dic, X_train, Y_train, X_val, Y_val, X_test, Y_test, maps)
trained_dic = deepcopy(dic)

if is_interactive():
    plt.subplots(figsize=(16,4*3))
    dic = trained_dic
    ax = plt.subplot2grid((4, 1), (0, 0), colspan=1)
    ax.plot(dic["acc"]["train"], lw=3, color="forestgreen", label="train acc")
    ax.plot(dic["acc"]["val"], lw=3, color="royalblue", label="val acc")
    m = min(np.min(dic["acc"]["val"]), np.min(dic["acc"]["train"]))
    M = max(np.max(dic["acc"]["val"]), np.max(dic["acc"]["train"]))
    ax.vlines(x=dic["best_acc"]["epoch"], ymin=m, ymax=1, lw=3, ls="--", color="royalblue", label="best acc")
    plt.legend()
    plt.grid()

    ax = plt.subplot2grid((4, 1), (1, 0), colspan=1)
    ax.plot(dic["mcc"]["train"], lw=3, color="forestgreen", label="train mcc")
    ax.plot(dic["mcc"]["val"], lw=3,  color="royalblue",label="val mcc")
    m = min(np.min(dic["mcc"]["val"]), np.min(dic["mcc"]["train"]))
    M = max(np.max(dic["mcc"]["val"]), np.max(dic["mcc"]["train"]))
    ax.vlines(x=dic["best_mcc"]["epoch"], ymin=m, ymax=M, lw=3, color="royalblue", ls="--", label="best mcc")
    plt.legend()
    plt.grid()

    ax = plt.subplot2grid((4, 1), (2, 0), colspan=1)
    ax.plot(dic["usage"]["train"], lw=3, color="forestgreen", label="train usage")
    ax.plot(dic["usage"]["val"], lw=3,  color="royalblue",label="val usage")
    m = min(np.min(dic["usage"]["val"]), np.min(dic["usage"]["train"]))
    M = max(np.max(dic["usage"]["val"]), np.max(dic["usage"]["train"]))
    ax.vlines(x=dic["best_usage"]["epoch"], ymin=m, ymax=M, lw=3, color="royalblue", ls="--", label="best usage")
    plt.legend()
    plt.grid()

    ax = plt.subplot2grid((4, 1), (3, 0), colspan=1)
    ax.plot(dic["loss"]["train"], lw=3, color="forestgreen", label="train loss")
    ax.plot(dic["loss"]["val"], lw=3, color="royalblue", label="val loss")
    ax.vlines(x=dic["best_loss"]["epoch"], ymin=0, ymax=np.max(dic["loss"]["val"]), lw=3, ls="--", color="royalblue", label="best loss")
    plt.legend()
    plt.grid()

if is_interactive():
    blue = '#4b4dce'
    green = '#117733'
    import matplotlib as mpl
    font = {'size'   : 30}
    mpl.rc('font', **font)

    plt.subplots(figsize=(16,9))
    plt.plot(np.array(dic["usage"]["train"])*100, lw=10, color=green, label="Training")
    plt.plot(np.array(dic["usage"]["val"])*100, lw=10,  color=blue,label="Validation")
    m = 100*min(np.min(dic["usage"]["val"]), np.min(dic["usage"]["train"]))
    M = 100*max(np.max(dic["usage"]["val"]), np.max(dic["usage"]["train"]))
    plt.vlines(x=dic["best_usage"]["epoch"], ymin=0, ymax=100, lw=3, color=blue, ls=":", label="Best Validation")
    plt.legend()
    plt.grid(axis='y')
    plt.ylim((0,100))
    plt.ylabel("Success rate (%)")
    plt.xlabel("Epochs")
    #plt.savefig("/home/tim/Sync/Figures/2022/02/23/example_NN_training.png")

# # Evaluate

config_test = deepcopy(config)
config_test["test_ratio"] = 1.
config_test["val_ratio"] = 0.

best_model = NN(config)
best_model.load_state_dict(trained_dic["best_usage"]["model"])  
best_model.cuda()
use_BCE = type(config['criterion']) == torch.nn.BCEWithLogitsLoss

if is_interactive():
    path = datapath + "saved_model/ALP_v4"
    #clean_save(path, config, best_model, standardization)

# ## Estimate best contact from training data

test_maps = []
X, Z = None, None
for dic in tqdm(maps["train"]):
    if X is None:
        X, Z = dic["X"], dic["Z"]
    test_maps.append(dic["map"])

# +
mean = np.mean(test_maps, axis=0)
iz_best, ix_best = np.unravel_index(np.argmax(mean), mean.shape)
x_best, z_best = X[ix_best], Z[iz_best]
training_best_score = np.max(mean)
print(training_best_score, (ix_best, iz_best), (x_best, z_best))

if is_interactive():
    fig, ax = plt.subplots(figsize=(9,9))
    mean = np.mean(test_maps, axis=0)
    line = ax.pcolor(mean, cmap="RdYlGn", edgecolors='k', vmin=0, vmax=1, linewidths=1)
    ax.set_xticks([i+0.5 for i in range(0,len(X))])
    ax.set_xticklabels([f"{x:1.2f}" for x in X], rotation=90)
    ax.set_yticks([i+0.5 for i in range(0,len(Z))])
    ax.set_yticklabels([f"{x:1.2f}" for x in Z])


# -

# ## Prediction

# +
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

def classify(truth, iz, ix, dic, classif):
    if truth[iz, ix]:
        classif["success"].append(dic)
    else:
        if np.sum(truth) == 0.:
            classif["impossible"].append(dic)
        else:
            classif["failure"].append(dic) 


# +
NN_res = {"success": [], "failure": [], "impossible": []}
constant_res = {"success": [], "failure": [], "impossible": []}

for dic in tqdm(maps["test"]):
    truth = dic["map"]
    samples = []
    X, Z = dic['X'], dic['Z']
    for x in X:
        for z in Z:
            samples.append(np.concatenate((np.copy(dic["input"]), [float(x)], [float(z)])))
    samples = torch.tensor(samples, dtype=torch.float)
    # Standardize
    if standardization is not None:
        samples = (samples - standardization['mean']) / standardization['std']
    # Infer 
    samples = samples.cuda()
    proba = compute_proba_pred(samples, model=best_model, use_BCE=use_BCE)
    k = 0
    dic["proba"] = proba
    proba_map = np.empty((len(Z), len(X)))
    for i, x in enumerate(X):
        for j, z in enumerate(Z):
            proba_map[j][i] = proba[k]
            k+=1
    dic["proba_map"] = proba_map
    
    z_index, x_index = np.unravel_index(np.argmax(proba_map), proba_map.shape)
    z_pred, x_pred = Z[z_index], X[x_index]
    dic["pred"] = {"x": x_pred, "z": z_pred, "z_index":z_index, "x_index": x_index, "proba": proba_map[z_index, x_index]}
    
    classify(truth, z_index, x_index, dic, NN_res)
    dic["NN_success"] = truth[z_index, x_index]
    classify(truth, iz_best, ix_best, dic, constant_res)
    dic["Constant_success"] = truth[iz_best, ix_best]
# -

# ## Estimate avoidable and robustly avoidable 

robustly_avoidable, avoidable = [], []
for dic in maps["test"]:
    T = dic["map"]
    min_pooled = np.zeros(T.shape)
    for z in range(len(T)):
        for x in range(len(T[z])):
            min_pooled[z,x] = np.min(T[max(0, z-1):z+2, max(0, x-1):x+2])
    if np.sum(min_pooled) > 0:
        robustly_avoidable.append(dic["key"])
        dic["robust_avoidable"] = True
    else:
        dic["robust_avoidable"] = False
    if np.sum(T) > 0:
        dic["avoidable"] = True
        avoidable.append(dic["key"])
    else:
        dic["avoidable"] = False

NN_res = {"avoided": [], "unavoided": [], "avoided among robust": [], "unavoided among robust": []}
Constant_res = {"avoided": [], "unavoided": [], "avoided among robust": [], "unavoided among robust": []}
keys = []
for dic in maps["test"]:
    keys.append(dic["key"])
    if dic["NN_success"]:
        NN_res["avoided"].append(dic["key"])
        if dic["robust_avoidable"]:
            NN_res["avoided among robust"].append(dic["key"])
    elif dic["avoidable"]:
        NN_res["unavoided"].append(dic["key"])
        if dic["robust_avoidable"]:
            NN_res["unavoided among robust"].append(dic["key"])
    if dic["Constant_success"]:
        Constant_res["avoided"].append(dic["key"])
        if dic["robust_avoidable"]:
            Constant_res["avoided among robust"].append(dic["key"])
    elif dic["avoidable"]:
        Constant_res["unavoided"].append(dic["key"])
        if dic["robust_avoidable"]:
            Constant_res["unavoided among robust"].append(dic["key"])

# +
print(config["damage"])
print(f"avoidable: {len(avoidable)/len(maps['test'])*100:2.1f}%") 
print(f'robustly avoidable among avoidable: {len(robustly_avoidable)/len(avoidable)*100:2.1f}%')

print(f"avoided among avoidable using the constant solution: {len(Constant_res['avoided'])/len(avoidable)*100:2.1f}%") 
print(f"avoided among robustly avoidable using the constant solution: {len(Constant_res['avoided among robust'])/len(robustly_avoidable)*100:2.1f}%") 

print(f"avoided among avoidable using NN: {len(NN_res['avoided'])/len(avoidable)*100:2.1f}%") 
print(f"avoided among robustly avoidable using NN: {len(NN_res['avoided among robust'])/len(robustly_avoidable)*100:2.1f}%") 
# -

# ### more fine analysis

with open(config_test['logdir']+"/evaluation.pk", "wb") as f:
    pickle.dump({"NN_res": NN_res, 
                 "Constant_res": Constant_res,
                 "keys": keys,
                 "avoidable": avoidable,
                 "robustly_avoidable": robustly_avoidable}, 
                f)

if is_interactive():
    with open(f"{datapath}datasets/{config['damage']}_dataset.pk", "rb") as f:
        testing_set = pickle.load(f)


# +
def add_dic(key, val, dic):
    if key in dic:
        dic[key].append(val)
    else:
        dic[key] = [val]
        
if is_interactive():
    robustly_avoidable, avoidable, unavoidable = {}, {}, {}
    for dic in maps["test"]:
        if dic["avoidable"]:
            if dic["robust_avoidable"]:
                add_dic(testing_set[dic["key"]]["no_reflex_end"], dic["NN_success"], robustly_avoidable)
            else:
                add_dic(testing_set[dic["key"]]["no_reflex_end"], dic["NN_success"], avoidable)
        else:
            add_dic(testing_set[dic["key"]]["no_reflex_end"], dic["pred"]["proba"], unavoidable)
# -

if is_interactive():
    print("Success rate:")
    print("\tRobustly avoidable:")
    for without_reflex in robustly_avoidable.keys():
        print(f"\t\t{without_reflex}: {100*np.mean(robustly_avoidable[without_reflex]):2.1f}% ({int(np.sum(robustly_avoidable[without_reflex]))}/{len(robustly_avoidable[without_reflex])})")
    print("\tUnrobustly avoidable:")
    for without_reflex in avoidable.keys():
        print(f"\t\t{without_reflex}: {100*np.mean(avoidable[without_reflex]):2.1f}% ({int(np.sum(avoidable[without_reflex]))}/{len(avoidable[without_reflex])})")

# #### Confidence

if is_interactive():
    data = np.random.random((10,3))
    sb.displot(data=unavoidable, kind="kde", fill=True, height=5, aspect=16/9, clip=(0,1))
    plt.xlabel("Confidence")

if is_interactive():
    fall_avoided = []
    fall_generated = []
    for τ in np.linspace(0,1,100):
        fg, fa = [], []
        for dic in maps["test"]:
            if dic["robust_avoidable"]:
                if testing_set[dic["key"]]["no_reflex_end"]:
                    fg.append((not dic["NN_success"]) and (dic["pred"]["proba"] > τ))
                else: 
                    fa.append(dic["NN_success"] and (dic["pred"]["proba"] > τ) )
        fall_generated.append(100*np.mean(fg))
        fall_avoided.append(100*np.mean(fa))

if is_interactive():
    plt.subplots(figsize=(10,6))
    X = np.linspace(0,1,100)
    plt.plot(X, fall_generated, lw=3, label="Fall generated")
    plt.plot(X, fall_avoided, lw=3, label="Fall avoided")
    plt.legend()
    plt.ylabel("%")
    plt.xlabel("Confidence treshold")

# ## Plot

# ### Examples

if is_interactive():
    res_maps = NN_res["unavoided among robust"]
    #res_maps = NN_res["avoided"]
    n = min(5, len(res_maps))
    plt.subplots(figsize=(4*n, 4*2))
    indices = rng.permutation(len(res_maps))
    for i in range(n):
        idx = indices[i]
        pred = res_maps[idx]["pred"]
        choosen = np.zeros(res_maps[idx]["proba_map"].shape)

        ax = plt.subplot2grid((2, n), (0, i), colspan=1)
        line = ax.pcolor(res_maps[idx]["truth"], cmap="viridis", edgecolors='k', vmin=0, vmax=1, linewidths=1)
        ax.scatter([pred["x_index"]+0.5],[pred["z_index"]+0.5], color="red",  s=100, marker="x")
        if i==0:
            ax.set_ylabel("Truth")
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        if i == n//2:
            plt.title("NN misspredictions among avoidable")

        ax = plt.subplot2grid((2, n), (1, i), colspan=1)
        line = ax.pcolor(res_maps[idx]["proba_map"], cmap="viridis", edgecolors='k', vmin=0, vmax=1, linewidths=1)

        ax.scatter([pred["x_index"]+0.5],[pred["z_index"]+0.5], color="red", s=100, marker="x", label="Selected contact")
        if i == n-1:
            plt.legend(bbox_to_anchor=(-2.55,1.23,1.3,0))
        if i==0:
            ax.set_ylabel("NN Prediction")
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
