import numpy as np


def dfunc(f, x):
    h = 1e-4  # 微小變化量，避免除以 0 發生
    grad = np.zeros_like(x)  # 初始化一個和 x 形狀相同，但所有元素都為0的陣列，用於存放梯度值

    # np.nditer是一個迭代器，用於迭代陣列 x 的每個元素，並能夠追踪當前元素的索引
    it = np.nditer(x, flags=["multi_index"])

    while not it.finished:  # 繼續迭代直到所有元素都被訪問
        idx = it.multi_index  # 取得當前元素的索引
        tmp_val = x[idx]  # 暫存當前元素的值

        # 計算 f(x+h)
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # 計算 f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        # 計算梯度（中心差分方法）
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 恢復原始值
        it.iternext()  # 移動到下一個元素

    return grad  # 返回梯度陣列
