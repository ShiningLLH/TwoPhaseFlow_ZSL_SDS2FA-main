import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import warnings
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from lightgbm import LGBMRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor

from networks import SiameseNetwork

warnings.filterwarnings('ignore')

# ================= 1. 环境配置 =================
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Device: {device}")

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================= 2. 数据加载函数 =================
def creat_dataset_problem(Task_index, test_index, attribute_matrix):
    attr_mat = attribute_matrix.values
    base_path = './GW_mat_data'

    if Task_index == 1:
        path = os.path.join(base_path, 'Dataset1')
        train_files = ['data_bubble.mat', 'data_plug.mat', 'data_slug.mat', 'data_wave.mat', 'data_st.mat',
                       'data_ann.mat']
        train_list = [loadmat(os.path.join(path, f))['data'] for f in train_files]
        train_data = np.vstack(train_list)
        train_attr = np.vstack([np.tile(attr_mat[i, :], (train_list[i].shape[0], 1)) for i in range(len(train_list))])

        test_files = ['data_b2s.mat', 'data_p2s.mat', 'data_s2a.mat', 'data_st2w.mat']
        test_list = [loadmat(os.path.join(path, f))['data'][500:700, :] for f in test_files]
        test_data = np.vstack(test_list)
        test_label = []
        for item in test_index:
            test_label += [item + 1] * (test_list[item - 6].shape[0])

    elif Task_index == 2:
        path = os.path.join(base_path, 'Dataset1')
        files = ['data_bubble.mat', 'data_plug.mat', 'data_slug.mat', 'data_wave.mat', 'data_st.mat', 'data_ann.mat']
        all_list = [loadmat(os.path.join(path, f))['data'] for f in files]
        train_idx = list(set(range(6)) - set(test_index))

        train_data = np.vstack([all_list[i] for i in train_idx])
        train_attr = np.vstack([np.tile(attr_mat[i, :], (all_list[i].shape[0], 1)) for i in train_idx])

        test_data = np.vstack([all_list[i][0:300] for i in test_index])
        test_label = []
        for i in test_index:
            test_label += [i + 1] * 300

    elif Task_index == 3:
        path = os.path.join(base_path, 'Dataset2')
        train_files = ['data_bubble.mat', 'data_plug.mat', 'data_slug.mat', 'data_wave.mat', 'data_st.mat',
                       'data_ann.mat']
        train_list = [loadmat(os.path.join(path, f))['data'] for f in train_files]
        train_data = np.vstack(train_list)
        train_attr = np.vstack([np.tile(attr_mat[i, :], (train_list[i].shape[0], 1)) for i in range(len(train_list))])

        test_files = ['data_1-1.mat', 'data_2-2.mat', 'data_1-6.mat', 'data_5-6.mat', 'data_5-1.mat', 'data_1-9.mat']
        test_list = [loadmat(os.path.join(path, f))['data'][0:300] for f in test_files]
        test_data = np.vstack(test_list)
        test_label = []
        for i in range(len(test_index)):
            test_label += [test_index[i] + 1] * test_list[i].shape[0]

    return train_data, train_attr, test_data, test_label


# ================= 3. 核心算法逻辑 =================
def pre_attribute_model(model_name, train_f, train_a, test_f):
    print(f'Attribute predictor: {model_name}')
    model_dict = {
        'Ridge': MultiOutputRegressor(Ridge(alpha=1)),
        'Lasso': MultiOutputRegressor(Lasso(alpha=0.1)),
        'rf': RandomForestClassifier(n_estimators=100),
        'lgbm': MultiOutputRegressor(LGBMRegressor(n_estimators=100, verbosity=-1)),
        'SVC_linear': MultiOutputRegressor(SVC(kernel='linear'))
    }
    clf = model_dict.get(model_name, MultiOutputRegressor(Ridge()))
    clf.fit(train_f, train_a)
    return clf.predict(test_f)


def pre_state_label(test_pre_attribute, attribute_matrix1, test_index):
    attribute_matrix = pd.DataFrame(attribute_matrix1.values)
    attribute_matrix2 = attribute_matrix.iloc[test_index, :]
    label_lis = []
    for i in range(test_pre_attribute.shape[0]):
        pre_res = test_pre_attribute[i, :]
        loc = (np.sum(np.square(attribute_matrix2.values - pre_res), axis=1)).argmin()
        label_lis.append(attribute_matrix2.index[loc] + 1)
    label_lis = np.array(np.row_stack(label_lis))
    return label_lis


def train_siamese_network(S2FANet, train_loader, criterion_attribute, optimizer, scheduler,
                          dimension_fea, epochs, para_steadiness, para_cov, para_error,
                          patience=20, min_delta=1e-4):
    # 将模型移动到计算设备
    S2FANet.to(device)
    I = torch.eye(dimension_fea).to(device)
    loss_history = []
    best_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        S2FANet.train()
        epoch_loss = 0.0
        for data in train_loader:
            X_data, Y_data = data[0].to(device), data[1].to(device)

            X1, Y1 = X_data[:-2], Y_data[:-2]
            X2 = X_data[1:-1]
            X3 = X_data[2:]

            output1, output2, output3, output_attri = S2FANet(X1, X2, X3)

            loss_error = criterion_attribute(output_attri, Y1)
            Cov = (output1.T @ output1) / (output1.shape[0] - 1)
            loss_cov = (Cov - I).norm(p="fro")
            loss_slowness = (output1 - output2).norm(p="fro")
            loss_steadiness = (output1 - 2 * output2 + output3).norm(p="fro")
            total_loss = loss_slowness + para_steadiness * loss_steadiness + para_cov * loss_cov + para_error * loss_error

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)
        scheduler.step()

        # 早停逻辑
        if avg_epoch_loss < best_loss - min_delta:
            best_loss = avg_epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"\nEarly stop： {epoch + 1} epoch")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {avg_epoch_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


def test_model(S2FANet, X_test, X_train):
    S2FANet.eval()
    X_test = X_test.to(device)
    X_train = X_train.to(device)
    with torch.no_grad():
        output1, _, _, _ = S2FANet(X_test, X_test, X_test)
        output1_train, _, _, _ = S2FANet(X_train, X_train, X_train)
        trainfeature = output1_train.cpu().numpy()  # 移回内存转numpy
        testfeature = output1.cpu().numpy()
    return trainfeature, testfeature


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    unique_classes = np.unique(y_true)
    class_counts = np.bincount(y_true)
    class_counts = class_counts[unique_classes]
    cm_normalized = cm.astype('float') / class_counts[:, np.newaxis]
    Average_accuracy = accuracy_score(y_true, y_pred)
    print("Average_accuracy:", Average_accuracy)

    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = cm_normalized.max() / 2.
    for i, j in ((i, j) for i in range(cm_normalized.shape[0]) for j in range(cm_normalized.shape[1])):
        plt.text(j, i, "{:.2f}".format(cm_normalized[i, j]),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()


# ================= 4. 主程序 =================
if __name__ == '__main__':
    set_random_seed(42)
    Task_index = 3

    if Task_index == 1:
        print("================= Task 1: ZSL for transition states =================")
        Fea_Attri_model = 'lgbm'
        attribute_matrix = pd.read_excel('./gw_attribute1.xlsx', index_col='no')
        test_index = [6, 7, 8, 9]
        classes = ['State 7', 'State 8', 'State 9', 'State 10']
    elif Task_index == 2:
        print("================= Task 2: ZSL for unknown states =================")
        Fea_Attri_model = 'lgbm'
        attribute_matrix = pd.read_excel('./gw_attribute.xlsx', index_col='no')
        test_index = [2, 3]
        classes = ['State 3', 'State 4']
    elif Task_index == 3:
        print("================= Task 3: ZSL for typical states under offset conditions =================")
        Fea_Attri_model = 'lgbm'
        attribute_matrix = pd.read_excel('./gw_attribute.xlsx', index_col='no')
        test_index = [0, 1, 2, 3, 4, 5]
        classes = ['State 1-2', 'State 2-2', 'State 3-2', 'State 4-2', 'State 5-2', 'State 6-2']

    # 数据预处理
    train_data, train_attributelabel, test_data, test_label = creat_dataset_problem(Task_index, test_index,
                                                                                    attribute_matrix)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # 训练配置
    dimension_fea, batch_size, lr, epochs = 10, 128, 0.01, 300
    lr_step_size, lr_gamma = 20, 0.8
    para_steadiness, para_cov, para_error = 2, 2, 2

    S2FANet = SiameseNetwork(dimension_fea)
    X_train = torch.FloatTensor(train_data)
    Y_train = torch.FloatTensor(train_attributelabel)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    criterion_attribute = nn.L1Loss()
    optimizer = optim.Adam(S2FANet.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)


    print("================= Training =================")
    train_siamese_network(S2FANet, train_loader, criterion_attribute, optimizer, scheduler,
                          dimension_fea, epochs, para_steadiness, para_cov, para_error,
                          patience=20, min_delta=1e-4)

    print("================= Testing =================")
    X_test = torch.FloatTensor(test_data)
    trainfeature, testfeature = test_model(S2FANet, X_test, X_train)

    test_pre_attribute = pre_attribute_model(Fea_Attri_model, trainfeature, train_attributelabel, testfeature)
    pre_label_lis = pre_state_label(test_pre_attribute, attribute_matrix, test_index)

    plot_confusion_matrix(test_label, pre_label_lis, classes)