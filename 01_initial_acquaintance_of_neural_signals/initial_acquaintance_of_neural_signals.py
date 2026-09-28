"""读取运动想象 EEG，观察连续波形和功率谱。

输入：MOABB 的 BNCI2014_001 数据，默认被试 1、0train 会话、第 0 个 run。
处理：选取 C5、C3、C1，查看前 10 秒波形及前 30 秒的 Welch 功率谱。
输出：打印数据概况；按需绘图，不返回或保存结果。
依赖：MOABB、MNE、Matplotlib。
"""

import matplotlib.pyplot as plt
import moabb
from moabb.datasets import BNCI2014_001


def main(
    show_plots: bool = True,
    subject: int = 1,
    session_name: str = "0train",
    run_name: str = "0",
) -> None:
    """读取指定记录，并按需显示波形和功率谱。"""
    # 1. 读取一个被试的记录
    moabb.set_log_level("info")
    dataset = BNCI2014_001()
    sessions = dataset.get_data(subjects=[subject])
    raw = sessions[subject][session_name][run_name]
    print(raw.info)

    # 2. 取出波形；MNE 内部使用伏特，绘图时换算为微伏。
    channels = ["C5", "C3", "C1"]
    sfreq = raw.info["sfreq"]
    end_sample = int(10 * sfreq)
    data_uv = raw.get_data(picks=channels, start=0, stop=end_sample) * 1e6
    times = raw.times[:end_sample]

    # 3. 在副本上截取前 30 秒，计算 4 Hz 至奈奎斯特频率的功率谱。
    raw_pick = raw.copy().pick(channels).crop(tmin=0, tmax=30)
    spectrum = raw_pick.compute_psd(
        method="welch",
        fmin=4.0,
        fmax=sfreq / 2,
    )

    if show_plots:
        fig, axes = plt.subplots(
            nrows=len(channels),
            ncols=1,
            sharex=True,
            figsize=(12, 6),
        )
        for ax, channel, signal_uv in zip(axes, channels, data_uv):
            ax.plot(times, signal_uv, linewidth=0.8)
            ax.set_ylabel(f"{channel}\n(μV)")
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Subject {subject} · {session_name} · run {run_name}")
        fig.tight_layout()
        spectrum.plot(show=False)
        plt.show(block=True)


if __name__ == "__main__":
    main()
