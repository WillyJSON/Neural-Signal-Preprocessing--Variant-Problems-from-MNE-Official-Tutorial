"""MNE 滤波与重采样教程的注释参考版。

输入：MNE sample 连续记录。
处理：1～40 Hz 带通、60 Hz 陷波，再重采样至 200 Hz。
输出：打印数据及采样率，按需绘制波形和功率谱。
依赖：MNE、Matplotlib；原学习说明保留在下方注释中。
"""

# MNE Filtering & Resampling 数据预处理流程（注释版）
# 
# 功能：
# 对连续神经电信号 Raw 数据进行：
# 1. 频率域清理（filtering）
# 2. 固定频率干扰去除（notch filtering）
# 3. 时间采样率调整（resampling）
# 
# 解决的问题：
# 原始神经信号通常包含：
# - 与研究无关的频率成分
# - 低频漂移
# - 高频噪声
# - 工频干扰
# 
# 这些成分会降低后续 spike detection、LFP分析、特征提取和机器学习建模效果。
# 
# 核心思想：
# 滤波：
#     决定“保留哪些频率”。
# 
# 重采样：
#     决定“用多少时间点表示信号”。
# 
# 数据流：
# Raw连续信号
#     ↓
# 频率清理后的Raw
#     ↓
# 采样率调整后的Raw

from pathlib import Path

import matplotlib.pyplot as plt
import mne

# 基于 MNE-Python Filtering and resampling data 教程。
# Copyright the MNE-Python contributors. SPDX-License-Identifier: BSD-3-Clause


def main(show_plots: bool = True, path: Path | None = None) -> None:
    # ============================================================
    # Block 1
    # 加载连续神经信号数据
    #
    # 输入：
    #   FIF格式原始记录文件
    #
    # 输出：
    #   raw
    #
    # raw内部：
    #   多通道时间序列 + 通道信息 + 采样率
    #
    # 直观理解：
    #   Raw就是一个保存脑电/脑磁连续波形的大容器。
    # ============================================================

    if path is None:
        path = Path(mne.datasets.sample.data_path()) / "MEG" / "sample"
        path = path / "sample_audvis_raw.fif"
    raw_fname = path


    raw = mne.io.read_raw_fif(
        raw_fname,
        preload=True
    )  # 读取FIF格式连续信号，并加载到内存


    print(raw)


    # ============================================================
    # Block 2
    # 查看原始信号频率组成
    #
    # 功能：
    #   在处理前观察数据中包含哪些频率。
    #
    # 输入：
    #   raw
    #
    # 输出：
    #   原始波形和频谱图
    #
    # 直观理解：
    #   滤波参数不是随便设置，需要先知道信号里面有什么。
    # ============================================================

    if show_plots:
        raw.plot(
            duration=10,
            n_channels=5
        )  # 显示部分通道的连续波形

    if show_plots:
        raw.compute_psd().plot()  # 计算功率谱密度，观察不同频率能量分布


    # ============================================================
    # Block 3
    # Band-pass滤波
    #
    # 功能：
    #   保留目标频率范围，删除其他频率。
    #
    # 输入：
    #   raw
    #
    # 输出：
    #   filtered_raw
    #
    # 例：
    #   1-40 Hz
    #   保留低频脑活动
    #   去除极慢漂移和高频噪声
    #
    # 直观理解：
    #   像频率筛子，只让需要的信息通过。
    # ============================================================

    filtered_raw = raw.copy()  # 复制数据，避免修改原始数据

    filtered_raw.filter(
        l_freq=1,
        h_freq=40
    )  # 高通1Hz + 低通40Hz，形成带通滤波


    # ============================================================
    # Block 4
    # Notch滤波
    #
    # 功能：
    #   去除固定频率干扰。
    #
    # 输入：
    #   filtered_raw
    #
    # 输出：
    #   notch_filtered_raw
    #
    # 常见原因：
    #   电网产生50Hz/60Hz噪声。
    #
    # 直观理解：
    #   在频谱中特定频率位置挖掉一个缺口。
    # ============================================================

    notch_filtered_raw = filtered_raw.copy()  # 复制滤波后的数据

    notch_filtered_raw.notch_filter(
        freqs=[60]
    )  # 去除60Hz附近工频噪声


    # ============================================================
    # Block 5
    # 重采样
    #
    # 功能：
    #   改变每秒采集点数量。
    #
    # 输入：
    #   notch_filtered_raw
    #
    # 输出：
    #   resampled_raw
    #
    # 注意：
    #   降采样前通常需要低通滤波，
    #   避免高频信号混入低频(aliasing)。
    #
    # 直观理解：
    #   1000Hz数据变成200Hz：
    #   数据量减少，但保留目标信息。
    # ============================================================

    resampled_raw = notch_filtered_raw.copy()  # 保留原始处理结果，避免覆盖

    resampled_raw.resample(
        sfreq=200
    )  # 将采样率调整为200Hz


    print(resampled_raw.info["sfreq"])  # 查看新的采样频率


    # ============================================================
    # Block 6
    # 检查处理结果
    #
    # 功能：
    #   比较处理前后的频谱变化。
    #
    # 输入：
    #   raw
    #   filtered_raw
    #   resampled_raw
    #
    # 输出：
    #   PSD图
    # ============================================================

    if show_plots:
        raw.compute_psd().plot()  # 原始频谱

    if show_plots:
        filtered_raw.compute_psd().plot()  # 滤波后频谱

    if show_plots:
        resampled_raw.compute_psd().plot()  # 重采样后频谱


    """
    工程迁移：

    MEA / EEG / BCI常见流程：

    Spike分析：
    Raw
     ↓
    高通滤波
     ↓
    Spike detection


    LFP分析：
    Raw
     ↓
    低通滤波
     ↓
    降采样
     ↓
    LFP特征分析


    机器学习：
    Raw
     ↓
    滤波
     ↓
    重采样
     ↓
    特征提取
     ↓
    模型输入


    关键理解：
    滤波改变频率信息。
    重采样改变时间分辨率。

    二者都是为了让后续分析得到更干净、更高效的数据。
    """

    if show_plots:
        plt.show(block=True)


if __name__ == "__main__":
    main()
