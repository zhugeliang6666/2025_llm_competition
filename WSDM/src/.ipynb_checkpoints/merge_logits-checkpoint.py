import sys
import pandas as pd
import pickle
import numpy as np
import numpy as np
import os


def softmax(x):
    """
    Compute the softmax of vector x.

    Parameters:
    x (numpy.ndarray): Input array or matrix.

    Returns:
    numpy.ndarray: Softmax of the input.
    """
    # Subtract the max value from each element to prevent overflow
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def softmax_to_logits(softmax_output):
    """
    从 softmax 输出还原 logits。

    参数:
    softmax_output (numpy.ndarray): softmax 输出的概率分布，二维数组。

    返回:
    numpy.ndarray: 还原的 logits，二维数组。
    """
    if softmax_output.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    # 取对数得到 log-probabilities
    log_probs = np.log(softmax_output)

    # 减去每行 log-probabilities 的最大值
    logits = log_probs - np.max(log_probs, axis=1, keepdims=True)

    return logits

if __name__ == '__main__':
    train_gemma_path = sys.argv[1]
    oof_qwen_path = sys.argv[2]
    save_path = sys.argv[3]
    with open(train_gemma_path,'rb') as f:
        data = pickle.load(f)
    with open(oof_qwen_path,'rb') as f:
        data_qwen = pickle.load(f)

    del data_qwen['text']
    data = data[['order_index','label','text']]

    data = data.merge(data_qwen,on=['order_index','label'])

    print(data)
    print(data.isnull().sum())
    with open(save_path,'wb') as f:
        pickle.dump(data,f)
    print(data.shape)
