import numpy as np

def kalman2D(list):
    dt = 1 / 30
    # print(dt)
    # 初始状态
    x_mat = np.mat([[list[0], ], [0, ], [0, ]])
    # 状态转移矩阵
    f_mat = np.mat([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]])
    # 初始协方差矩阵
    p_mat = np.mat([[0.01, 0, 0], [0, 0.0001, 0], [0, 0, 0.0001]])
    # 状态转移协方差
    q_mat = np.mat([[1, 0, 0], [0, 0.1, 0], [0, 0, 0.01]])
    # 定义观测矩阵
    h_mat = np.mat([1, 0, 0])
    # 定义观测噪声协方差
    r_mat = np.mat([20])
    predicts = [list[0]]
    for i in range(1, len(list)):
        '''
        x_predict = f_mat * x_mat
        p_predict = f_mat * p_mat * f_mat.T + q_mat
        kalman = p_predict * h_mat.T / (h_mat * p_predict * h_mat.T + r_mat)
        x_mat = x_predict + kalman * (list[i] - h_mat * x_predict)
        p_mat = (np.eye(3) - kalman * h_mat) * p_predict
        predicts.append(x_predict[0,0])
        '''
        x_predict = np.dot(f_mat, x_mat)
        p_predict = f_mat * p_mat * f_mat.T + q_mat
        kalman = p_predict * h_mat.T / (h_mat * p_predict * h_mat.T + r_mat)
        x_mat = x_predict + kalman * (list[i] - h_mat * x_predict)
        p_mat = (np.eye(3) - kalman * h_mat) * p_predict
        predicts.append(x_predict[0, 0])

    return predicts


def kalman1D(list):
    predicts = [list[0]]
    position_predict = predicts[0]
    predict_var = 0
    v_std = 1  # yu ce fang cha
    odo_var = 100  # ce liang fang cha
    for i in range(1, len(list)):
        predict_var += v_std ** 2
        position_predict = position_predict * odo_var / (predict_var + odo_var) + list[i] * predict_var / (
                predict_var + odo_var)
        predict_var = (predict_var * odo_var) / (predict_var + odo_var) ** 2
        predicts.append(position_predict)
    return predicts