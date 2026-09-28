"""观察高通与陷波滤波对 MEG 波形和频谱的影响。

输入：MNE sample 数据的前 120 秒，保留磁强计和刺激通道。
处理：0.33 Hz 高通后，对 60、120、180 Hz 陷波，再取前 60 秒；不重采样。
输出：按需显示波形及陷波前后功率谱，不返回或保存结果。
依赖：MNE、NumPy、Matplotlib。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

# 基于 MNE-Python Filtering and resampling data 教程。
# Copyright the MNE-Python contributors. SPDX-License-Identifier: BSD-3-Clause


def main(show_plots: bool = True, path: Path | None = None) -> None:
    """读取示例数据，依次高通、陷波并比较结果。"""
    # 1. 先裁剪、选道，再加载波形，避免读入整个记录。
    if path is None:
        path = Path(mne.datasets.sample.data_path()) / "MEG" / "sample"
        path = path / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(path, preload=False)
    # 窄过渡带产生较长 FIR，先保留上下文，滤波后再裁剪对照片段。
    raw.crop(tmin=0, tmax=120).pick(["mag", "stim"]).load_data()

    # 2. 高通滤波；保留原始副本供波形对照。
    raw_filt = raw.copy().filter(l_freq=0.33, h_freq=None)
    picks = mne.pick_types(raw_filt.info, meg=True, exclude="bads")

    # 3. 在独立副本上去除工频及谐波，不覆盖高通结果。
    notch_raw_filt = raw_filt.copy().notch_filter(
        freqs=[60, 120, 180],
        picks=picks,
        notch_widths=np.array([0.2, 0.23, 0.33]),
        trans_bandwidth=0.1,
    )

    for data in (raw, raw_filt, notch_raw_filt):
        data.crop(tmax=60)

    if show_plots:
        with mne.viz.use_browser_backend("matplotlib"):
            for title, data in (("Original", raw), ("High-pass 0.33 Hz", raw_filt)):
                data.plot(
                    title=title,
                    duration=60,
                    n_channels=len(data.ch_names),
                    proj=False,
                    remove_dc=False,
                    show=False,
                )

        # Welch 默认先平均各时间段；plot 的 average=True 再平均通道。
        for title, data in (
            ("Before notch", raw_filt),
            ("After notch", notch_raw_filt),
        ):
            spectrum = data.compute_psd(
                method="welch",
                fmin=0,
                fmax=250,
                picks=picks,
            )
            fig = spectrum.plot(
                average=True,
                amplitude=False,
                exclude="bads",
                show=False,
            )
            fig.suptitle(title)
        plt.show(block=True)


if __name__ == "__main__":
    main()
