"""比较坏道标记及额外标坏正常通道对 EEG 插值的影响。

输入：MNE sample 数据，保留已有坏道名单及电极位置。
处理：对前 3 秒 EEG 比较原波形、原名单插值和额外标坏后的插值。
输出：打印坏道名单，按需绘图；可选查看标记在 Epochs/Evoked 中的传递。
依赖：MNE、Matplotlib。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne

# 基于 MNE-Python Handling bad channels 教程。
# Copyright the MNE-Python contributors. SPDX-License-Identifier: BSD-3-Clause


def main(
    show_plots: bool = True,
    path: Path | None = None,
    compare_evoked: bool = False,
) -> None:
    """比较两份坏道名单的插值结果，可选执行原有事件平均示例。"""
    # 1. 读取数据；额外标坏不覆盖文件已有的名单。
    if path is None:
        path = Path(mne.datasets.sample.data_path()) / "MEG" / "sample"
        path = path / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
    raw_overmarked = raw.copy()
    raw_overmarked.info["bads"] = list(
        dict.fromkeys(raw.info["bads"] + ["EEG 051", "EEG 052", "EEG 053"])
    )
    print("Original bads:", raw.info["bads"])
    print("Overmarked bads:", raw_overmarked.info["bads"])

    # 2. 插值前保留坏道位置，reset_bads=False 保留标签供对照。
    eeg_short = raw.copy().crop(0.0, 3.0).pick("eeg", exclude=[]).load_data()
    eeg_short_interp = eeg_short.copy().interpolate_bads(
        reset_bads=False,
        origin="auto",
        method={"eeg": "spline"},
    )
    eeg_short_overmarked = (
        raw_overmarked.copy().crop(0.0, 3.0).pick("eeg", exclude=[]).load_data()
    )
    eeg_short_interp_overmarked = eeg_short_overmarked.copy().interpolate_bads(
        reset_bads=False,
        origin="auto",
        method={"eeg": "spline"},
    )

    # 3. 原来注释中的分段与平均对照，按参数选择是否执行。
    evokeds = {}
    if compare_evoked:
        raw_unmarked = raw.copy()
        raw_unmarked.info["bads"] = []
        events = mne.find_events(raw, stim_channel="STI 014", verbose=False)
        for condition, data in (
            ("Unmarked", raw_unmarked),
            ("Original bads", raw),
            ("Overmarked", raw_overmarked),
        ):
            picks = mne.pick_types(data.info, meg=True, eeg=True, exclude=[])
            epochs = mne.Epochs(
                data,
                events,
                event_id={"auditory/right": 2},
                tmin=-0.3,
                tmax=0.5,
                picks=picks,
                proj=False,
                flat=None,
                reject=None,
                preload=True,
                verbose=False,
            )
            evoked = epochs.average(picks=epochs.ch_names)
            evokeds[condition] = evoked
            print(condition, "Epochs bads:", epochs.info["bads"])
            print(condition, "Evoked bads:", evoked.info["bads"])

    # 4. 固定同一尺度和片段，对照 EEG 050～059。
    if show_plots:
        with mne.viz.use_browser_backend("matplotlib"):
            for title, data in (
                ("Before interpolation", eeg_short),
                ("After interpolation", eeg_short_interp),
                ("After interpolation with extra bads", eeg_short_interp_overmarked),
            ):
                picks = mne.pick_channels_regexp(data.ch_names, regexp="EEG 05.")
                data.plot(
                    title=title,
                    picks=picks,
                    n_channels=len(picks),
                    duration=3.0,
                    scalings={"eeg": 20e-6},
                    proj=False,
                    remove_dc=False,
                    show=False,
                )
        for condition, evoked in evokeds.items():
            fig = evoked.plot(proj=False, exclude="bads", show=False)
            fig.suptitle(f"{condition}: evoked response", fontsize=14)
        plt.show(block=True)


if __name__ == "__main__":
    main()
