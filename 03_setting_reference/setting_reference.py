"""比较 EEG 原始参考与 REST，并保留平均参考顺序的可选实验。

输入：MNE sample 前 60 秒的 EEG 041～059。
处理：补回参考通道，建立球模型与前向模型后执行 REST；可选比较删道顺序。
输出：按需显示参考前后波形，不返回或保存结果。
依赖：MNE、Matplotlib。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne

# 基于 MNE-Python Setting the EEG reference 教程。
# Copyright the MNE-Python contributors. SPDX-License-Identifier: BSD-3-Clause


def main(
    show_plots: bool = True,
    path: Path | None = None,
    compare_average: bool = False,
) -> None:
    """执行 REST 对照；compare_average 开启原有的平均参考顺序示例。"""
    # 1. 准备 EEG；保留文件中的坏道标记。
    if path is None:
        path = Path(mne.datasets.sample.data_path()) / "MEG" / "sample"
        path = path / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
    channels = [f"EEG {number:03d}" for number in range(41, 60)]
    raw.crop(tmax=60).pick(channels).load_data()

    # 去掉未应用的 SSP，避免将额外投影混入参考方法的比较。
    raw.del_proj()
    raw_new_ref = mne.add_reference_channels(raw, ref_channels="EEG 999")

    # 2. 原来注释中的三个平均参考方案，按参数选择是否执行。
    average_cases = []
    if compare_average:
        drop_names = [f"EEG {number:03d}" for number in range(41, 50)]
        keep_names = [ch for ch in raw_new_ref.ch_names if ch not in drop_names]

        raw_drop_ref = raw_new_ref.copy().pick(keep_names)
        raw_drop_ref.set_eeg_reference(ref_channels="average")

        raw_drop_after_ref = raw_new_ref.copy()
        raw_drop_after_ref.set_eeg_reference(ref_channels="average")
        raw_drop_after_ref.pick(keep_names)

        raw_drop_proj_ref = raw_new_ref.copy()
        raw_drop_proj_ref.set_eeg_reference(ref_channels="average", projection=True)
        raw_drop_proj_ref.pick(keep_names).apply_proj()
        average_cases = [
            ("Drop then average", raw_drop_ref),
            ("Average then drop", raw_drop_after_ref),
            ("Drop then apply projection", raw_drop_proj_ref),
        ]

    # 3. REST：球模型、源网格和前向模型均使用同一组 EEG 坐标。
    sphere = mne.make_sphere_model("auto", "auto", raw.info)
    src = mne.setup_volume_source_space(sphere=sphere, exclude=30.0, pos=15.0)
    forward = mne.make_forward_solution(
        raw.info,
        trans=None,
        src=src,
        bem=sphere,
        meg=False,
        eeg=True,
    )
    raw_rest = raw.copy().set_eeg_reference(ref_channels="REST", forward=forward)

    # 4. 标题与数据一一对应；关闭显示投影以比较实际波形。
    if show_plots:
        with mne.viz.use_browser_backend("matplotlib"):
            for title, data in [("Original", raw), ("REST", raw_rest)] + average_cases:
                data.plot(
                    title=f"{title} reference",
                    scalings={"eeg": 5e-5},
                    n_channels=len(data.ch_names),
                    proj=False,
                    remove_dc=False,
                    show=False,
                )
        plt.show(block=True)


if __name__ == "__main__":
    main()
