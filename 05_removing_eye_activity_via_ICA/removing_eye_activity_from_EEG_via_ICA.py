"""使用 ICA 去除 EEG 中的眼动伪迹。

输入：同目录 A01T.mat 的第 4 个 run，含 22 路 EEG、3 路 EOG 和采样率。
处理：将微伏转换为伏特，在 1 Hz 高通滤波副本上拟合 Picard ICA，
      再按指定排除成分，在未滤波的原始数据副本上重建信号。
输出：打印 ICA 成分数，显示诊断图及处理前后波形；不返回或保存结果。
依赖：MNE >= 1.13、NumPy、SciPy、Matplotlib、python-picard。

其中 A01T.mat 来自 https://bbci.de/competition/iv/download/index.html?agree=yes&submit=Submit 的 BCI Competition IV Dataset 2a
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.io import loadmat


FNAME = "A01T.mat"
DATA_PATH = Path(__file__).resolve().parent / FNAME

# 通道顺序必须与 X 的 25 列一致：前 22 列 EEG，后 3 列 EOG。
CH_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
    "EOG1", "EOG2", "EOG3",
]

# 手动排除的 ICA 成分，索引从 0 开始。
EXCLUDE_COMPONENTS = [1]


def main(path: Path = DATA_PATH, show_plots: bool = True) -> None:
    """读取数据、拟合 ICA，并在原始信号副本上去除指定成分。"""
    # 1. 读取数据
    if not path.is_file():
        raise FileNotFoundError(f"未找到数据文件：{path}")

    mat = loadmat(path, simplify_cells=True)
    run = mat["data"][3]  # 第 4 个 run
    signal_uv = np.array(run["X"], dtype=np.float64)

    if not np.isfinite(signal_uv).all():
        raise ValueError("输入信号包含 NaN 或 Inf，请先处理无效值。")

    # 微伏转为伏特，并转为 MNE 所需的 (通道数, 采样点数)。
    signal_v = (signal_uv * 1e-6).T
    sfreq = run["fs"]

    # 2. 构建 Raw 对象
    ch_types = ["eeg"] * 22 + ["eog"] * 3
    raw_info = mne.create_info(
        ch_names=CH_NAMES,
        sfreq=sfreq,
        ch_types=ch_types,
    )
    raw = mne.io.RawArray(signal_v, raw_info)
    raw.set_montage("easycap-M1")

    eeg_picks = mne.pick_types(raw_info, eeg=True, exclude="bads")

    # 3. 在副本上高通滤波，减轻低频漂移对 ICA 拟合的影响。
    raw_filt = raw.copy()
    raw_filt.filter(l_freq=1.0, h_freq=None)

    # 4. 拟合 ICA
    ica = mne.preprocessing.ICA(
        n_components=None,
        method="picard",
        fit_params={
            "extended": True,
            "ortho": False,
        },
        rng=1,  # 需要 MNE 1.13 或更高版本
    )
    ica.fit(raw_filt, picks=eeg_picks)
    print(f"ICA 成分数：{ica.n_components_}")

    # 5. 指定并检查成分
    ica.exclude = EXCLUDE_COMPONENTS

    if show_plots:
        # 用 EOG 相关性辅助检查，不自动修改排除列表。
        eog_name = "EOG1"
        candidates, scores = ica.find_bads_eog(
            raw_filt,
            ch_name=eog_name,
            measure="correlation",
            threshold=0.5,
        )
        ica.plot_scores(scores, exclude=candidates)

        mne.viz.set_browser_backend("matplotlib")
        ica.plot_components(
            inst=raw_filt,
            picks=range(ica.n_components_),
            show=False,
        )

        if ica.exclude:
            # 对比成分 1、7、8 的特征；这里只查看，不设置排除。
            ica.plot_properties(
                inst=raw_filt,
                picks=[1, 7, 8],
                show=False,
            )

        plt.show(block=True)

        ica.plot_sources(inst=raw_filt, block=True)

    # 6. 在未滤波的原始信号副本上重建
    raw_corrected = raw.copy()
    ica.apply(raw_corrected)

    # 7. 对比处理前后的波形
    if show_plots:
        raw.plot(
            title="Before ICA",
            proj=False,
            remove_dc=False,
            show=False,
            verbose=False,
        )
        raw_corrected.plot(
            title="After ICA",
            proj=False,
            remove_dc=False,
            show=False,
            verbose=False,
        )
        plt.show(block=True)


if __name__ == "__main__":
    main()
