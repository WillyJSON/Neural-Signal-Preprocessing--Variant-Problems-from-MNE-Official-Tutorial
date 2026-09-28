"""MNE 坏道处理教程的注释参考版。

输入：MNE sample 连续记录及已有坏道标记。
处理：检查标记和选道规则，分段平均，并演示 EEG/MEG 插值。
输出：打印名单与数据概况，按需绘图；迁移任务仍为注释。
依赖：MNE、NumPy、Matplotlib。
"""

# -*- coding: utf-8 -*-
# ============================================================================
# MNE 坏道处理：识别、标记、排除与插值（中文注释学习版）
# ============================================================================
# 参考：MNE-Python 官方教程 Handling bad channels
# https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html
# 原教程代码作者：The MNE-Python contributors；原代码采用 BSD-3-Clause 许可。
# 本文件保留教程的核心操作，重新组织功能板块，并补充参数解释和迁移任务。
# 运行核验：MNE 1.13.2，使用官方截短 sample 记录验证数值流程与绘图创建。
# 核验未覆盖首次完整数据下载和鼠标交互；原始 sample 数据的采样率、时长以实际文件为准。
#
# 【功能与问题】
# 电极接触不良或传感器异常，可能产生平直、过大或异常嘈杂的信号。
# 本文件演示如何检查并记录这些通道，控制它们是否参与后续操作，
# 以及在需要维持通道布局时，用正常传感器的信息重建坏道。
# 这是人工检查与处理教程，不包含自动坏道检测算法。
#
# 【核心机制】
# 1. 标记：把通道名称写入 info['bads']；只改元信息，不改信号数值。
# 2. 排除：根据坏道名单选择参与某项操作的通道；具体行为由该 API 决定。
# 3. 插值：依据正常传感器的信号与空间几何，计算坏道位置的估计信号。
#    EEG 示例使用球面样条；MEG 示例使用基于场映射的方法。
#    插值保留通道数量，但不创造与正常通道独立的新测量信息。
#
# 【板块架构】
# 01 读取示例数据与已有坏道记录
# 02 浏览异常通道
# 03 编辑与交互标记坏道
# 04 验证坏道的排除规则
# 05 检查坏道记录在分段与平均中的传递
# 06 插值修复 EEG 并比较波形
# 07 插值修复 MEG 梯度计并比较波形
# 文件末尾：迁移任务——临时隐藏正常 EEG 通道，评估插值误差。
#
# 【运行与学习方式】
# 安装依赖：python -m pip install mne numpy matplotlib
# 运行文件：python handling_bad_channels_annotated.py
# 原有板块注释保留用于阅读；调用 main() 执行整个流程。
# 使用官方 sample 数据集；首次运行若无本地缓存，data_path() 会尝试下载。
# 数据集下载体积明显大于本例实际读取的片段，请等待下载完成。
# show_plots 控制是否生成图；interactive_marking 控制是否暂停进行人工标记。
# 交互标记需要支持交互窗口的 Matplotlib 环境；静态 inline 图不能替代它。
# 图标题使用英文，避免本机未安装中文字体时显示方框。
# 本文件不覆盖原始 FIF 文件；大部分处理作用在副本上。
#
# 【对象与维度约定】
# Raw：连续记录对象；取出数据时为 ndarray，形状 (通道数, 采样点数)。
# Epochs：分段对象；取出数据时为 ndarray，形状 (试次数, 通道数, 段内采样点数)。
# Evoked：跨试次平均响应；.data 为 ndarray，形状 (通道数, 段内采样点数)。
# info['bads']：list[str]，存的是精确通道名称，不是整数索引。
# MNE 内部 EEG 单位为 V，MEG 磁强计为 T，梯度计为 T/m；绘图会转换显示单位。


# %% 01 读取示例数据与已有坏道记录
# 功能与问题：建立后续分析对象，同时保留采集文件中已记录的坏道信息。
# 实现机制：定位 sample 数据集，读取 FIF 头信息，暂不把完整波形载入内存。
# 输入：磁盘上的 sample_audvis_raw.fif，包含信号、事件通道及传感器几何。
# 输出：raw（mne.io.Raw）；raw.info（mne.Info）；file_bads（list[str]）。
# 关键 API：mne.datasets.sample.data_path()、mne.io.read_raw_fif()、raw.info。


from copy import deepcopy  # 复制可变列表，避免备份与原列表指向同一对象。
from pathlib import Path  # 用路径对象拼接目录，兼容 Windows 与 Linux。
import matplotlib.pyplot as plt
import numpy as np
import mne


def main(
    show_plots: bool = True,
    path: Path | None = None,
    interactive_marking: bool = False,
) -> None:
    if path is None:
        path = Path(mne.datasets.sample.data_path()) / "MEG" / "sample"
        path = path / "sample_audvis_raw.fif"
    raw_path = path
    raw = mne.io.read_raw_fif(
        raw_path,
        preload=False,  # 延迟读取波形；后续插值会在短片段上显式 load_data()。
        verbose=False,  # 只减少本次读文件的日志，不改变数据处理行为。
    )
    file_bads = deepcopy(raw.info["bads"])  # 保存文件最初的坏道名单，供后面核对。
    print("MNE version:", mne.__version__)
    print("Raw dimensions:", len(raw.ch_names), "channels x", raw.n_times, "samples")
    print("Sampling frequency (Hz):", raw.info["sfreq"])
    print("Bad channels stored in file:", file_bads)  # 官方示例为 ['MEG 2443', 'EEG 053']。


    # %% 02 浏览异常通道
    # 功能与问题：观察可疑通道相对同类型通道的异常，建立人工判断依据。
    # 实现机制：用正则表达式选择一组通道，在相同类型的显示尺度下比较波形。
    # 输入：raw；raw.ch_names 为按记录顺序排列的 list[str]。
    # 输出：eeg_view_picks、grad_view_picks（list[int]，零起始索引）；波形窗口。
    #       此处绘制副本，误点击图中的通道不会改变主对象 raw 的名单。
    # 关键 API：mne.pick_channels_regexp()、Raw.copy()、Raw.plot()、
    #           mne.viz.use_browser_backend()。
    # 判断提示：观察持续平直、异常幅度和异常噪声；单次振幅高不能独立证明坏道。
    # 通道编号相近不保证空间相邻，实际空间位置应以传感器坐标为准。

    eeg_view_picks = mne.pick_channels_regexp(
        raw.ch_names,
        regexp="EEG 05.",  # 正则中的 . 匹配任意单个字符；在本数据中选 EEG 050～059。
    )
    grad_view_picks = mne.pick_channels_regexp(
        raw.ch_names,
        regexp="MEG 2..3",  # 匹配 MEG 2 开头、末位为 3 的这组名称；此命名约定属本设备。
    )
    print("EEG channels for inspection:", [raw.ch_names[i] for i in eeg_view_picks])

    if show_plots:
        # 指定 MNE 浏览器后端，后续可用 Matplotlib Figure 方法。
        with mne.viz.use_browser_backend("matplotlib"):
            for title, picks in (
                ("Inspect EEG", eeg_view_picks),
                ("Inspect gradiometers", grad_view_picks),
            ):
                raw.copy().plot(
                    order=picks,  # 显示这些整数索引对应的通道；不会从 raw 中删除其他通道。
                    n_channels=len(picks),  # 同时显示所选通道，避免只显示默认的前几条。
                    duration=10.0,  # 窗口初始时间跨度，单位秒。
                    proj=False,  # 不在显示时启用尚未应用的 SSP 投影，以便查看异常。
                    bad_color="r",  # 已标坏通道显示为红色。
                    theme="light",  # 固定浅色主题，让黑色波形与背景保持清晰对比。
                    title=title,
                    show=False,  # 先创建图，文件末尾统一显示；手动标记入口另设在板块 03。
                )


    # %% 03 编辑与交互标记坏道
    # 功能与问题：把检查结论变成后续处理可使用的名单，并避免误覆盖已有记录。
    # 实现机制：先在副本演示 Python 列表增删；需要人工确认时，在主 raw 上交互修改。
    # 输入：raw 与已有 info['bads']（list[str]）。
    # 输出：用于列表练习的 raw_edit；最终用于后续处理的 raw.info['bads']。
    #       信号数值、通道总数和采样点数均不因“标记”本身发生变化。
    # 关键 API：deepcopy()、list.append()、list.extend()、list.pop()、
    #           Raw.plot(block=True)、raw.info['bads']。

    raw_edit = raw.copy()  # 列表练习独立进行，不把随意选出的练习通道带入正式流程。
    original_bads = deepcopy(raw_edit.info["bads"])  # 此处元素都是字符串，用 .copy() 也足够。
    raw_edit.info["bads"].append("EEG 050")  # append 添加一个名称；不要把多个名称组成的列表传给它。
    raw_edit.info["bads"].extend(["EEG 051", "EEG 052"])  # extend 逐项添加；添加前应避免已有名称重复。
    removed_bad = raw_edit.info["bads"].pop(-1)  # -1 指最后一项；返回被移除的字符串，空列表会报错。
    print("Removed from demonstration list:", removed_bad)
    print("Demonstration list:", raw_edit.info["bads"])
    raw_edit.info["bads"] = original_bads.copy()  # 这是整表替换；真实工作中不可无意覆盖已有坏道。

    if show_plots and interactive_marking:
        with mne.viz.use_browser_backend("matplotlib"):
            raw.plot(
                n_channels=40,  # 只控制同屏数量，其余通道可在浏览器中滚动查看。
                duration=10.0,
                proj=False,
                bad_color="r",
                theme="light",
                title="Click channel names to toggle bads; close to continue",
                block=True,  # 关键：暂停脚本，关窗后才执行后续分析，防止读取尚未确认的名单。
            )
        # 点击通道名或 Raw 波形可以切换好/坏状态，关闭窗口后修改保留在 raw 中。

    analysis_bads = raw.info["bads"].copy()  # 记录实际采用的名单，便于以后追踪哪些通道被处理过。
    print("Bad channels used in later steps:", analysis_bads)
    # 若需保存人工标记，可另行运行 raw.save('marked_raw.fif', overwrite=False)。
    # 保存 FIF 会保留 bads；只执行上面的列表操作并不会改写磁盘文件。


    # %% 04 验证坏道的排除规则
    # 功能与问题：确认某项操作到底使用了哪些通道，避免把“已标记”等同于“处处自动排除”。
    # 实现机制：分别获取全部 EEG 和正常 EEG 的索引，用集合差找到被排除的通道。
    # 输入：raw.info；raw.ch_names（list[str]）。
    # 输出：all_eeg_picks、good_eeg_picks、excluded_eeg_picks（1D 整数 ndarray）；
    #       eeg_all、eeg_good（Raw 子集），二者仍为连续记录对象。
    # 关键 API：mne.pick_types()、np.setdiff1d()、Raw.pick()。

    # 默认 exclude='bads'，排除标坏 EEG。
    good_eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True)
    # 空排除表：保留标坏 EEG。
    all_eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])
    # 返回前者有、后者没有的索引，并排序去重。
    excluded_eeg_picks = np.setdiff1d(all_eeg_picks, good_eeg_picks)
    # 转为 ndarray 后才能用整数数组批量索引。
    excluded_eeg_names = np.array(raw.ch_names)[excluded_eeg_picks]
    print("Excluded EEG indices:", excluded_eeg_picks)
    print("Excluded EEG names:", excluded_eeg_names)

    # Raw.pick 默认 exclude=()，通常保留坏道；此处显式写出。
    eeg_all = raw.copy().pick("eeg", exclude=[])
    eeg_good = raw.copy().pick("eeg", exclude="bads")  # 按类型选道并排除坏道，只改变这个副本。
    print(
        'EEG count with / without bads:',
        len(eeg_all.ch_names),
        '/',
        len(eeg_good.ch_names),
    )
    # 上面的 mne.pick_types 返回索引，不修改 raw；Raw.pick 则原地修改它所作用的对象。
    # exclude 对按类型选道生效；若显式传入通道名称或索引，坏道仍可被选中。
    # 插值之前需要保留待修复通道，因此板块 06 会再次显式选取全部 EEG。


    # %% 05 检查坏道记录在分段与平均中的传递
    # 功能与问题：展示遗漏坏道标记会让异常通道进入结果图，并确认派生对象保留坏道记录。
    # 实现机制：比较“清空标记”和“保留标记”两条分支；按事件分段，再对同类试次求平均。
    # 输入：raw；STI 014 中的触发信号。
    # 中间：events 为整数 ndarray，形状 (事件数, 3)，每行是
    #       [事件采样点编号, 变化前的事件值, 变化后的事件值]。
    #       epochs 为 Epochs，逻辑数据形状 (试次数, MEG+EEG 通道数, 段内采样点数)。
    # 输出：evokeds（dict[str, Evoked]），各 Evoked.data 为 (通道数, 段内采样点数)。
    # 关键 API：mne.find_events()、mne.Epochs()、Epochs.average()、Evoked.plot()。
    # 本块显式关闭 SSP；proj=False 不能撤销已写入数据的投影、ICA 或重参考处理。
    # 原教程以默认参数构造 Epochs，本文件显式指定 proj=False，使操作符合其检查建议。

    raw_unmarked = raw.copy()
    raw_unmarked.info["bads"] = []  # 仅作“漏标”对照；清空名单不代表原来的坏道恢复正常。
    # 从触发通道的数值变化提取事件。
    events = mne.find_events(raw, stim_channel="STI 014", verbose=False)
    print("Events shape:", events.shape)
    evokeds = {}

    for condition, raw_case in (("Unmarked", raw_unmarked), ("Marked", raw)):
        # 显式保留坏道以观察标记传递。
        meeg_picks = mne.pick_types(raw_case.info, meg=True, eeg=True, exclude=[])
        epochs = mne.Epochs(
            raw_case,
            events=events,
            event_id={"auditory/right": 2},  # 本 sample 数据中事件码 2 为右耳听觉刺激；不是数组第 2 个事件。
            tmin=-0.2,  # 每段从事件前 0.2 秒开始。
            tmax=0.5,  # 到事件后 0.5 秒结束；时间换算遵循原始采样率。
            baseline=(None, 0),  # 每通道减去段开始至 0 秒的均值；这是基线校正，不是跨通道重参考。
            picks=meeg_picks,  # 使用 raw_case 中的整数索引，显式包括已标坏的 MEG/EEG。
            proj=False,  # Epochs 默认会应用 SSP；此处关闭，防止检查时掩盖异常。
            reject=None,  # 不额外按峰峰值阈值剔除试次；不代表越界或 BAD 注释对应试次一定保留。
            flat=None,  # 不额外按平直阈值剔除试次。
            preload=True,  # 将所选事件片段载入内存，之后平均时无需反复读文件。
            verbose=False,
        )
        evoked = epochs.average(picks=epochs.ch_names)  # 显式传名称，保留坏道及其标签，便于检查传递。
        evokeds[condition] = evoked  # 保存 Evoked 对象；.plot() 返回的是图，不能用它代替 Epochs/Evoked。
        print(condition, "Epochs bads:", epochs.info["bads"])
        print(
            condition,
            'Evoked bads:',
            evoked.info['bads'],
            '; data shape:',
            evoked.data.shape,
        )
        if show_plots:
            fig = evoked.plot(exclude="bads", proj=False, show=False)  # 此处排除坏道是在绘图阶段生效。
            # Evoked 图采用自身布局，不强行 subplots_adjust。
            fig.suptitle(f"{condition}: evoked response", fontsize=14)

    # 这个对照中，单通道的跨试次平均不会混入其他通道的数值。
    # 标记后的图不再显示坏道，不能据此说“其他正常通道已经被修复”。
    # 新创建的对象继承当时的坏道名单；之后再修改 raw，不会追溯同步已有的 epochs/evoked。
    # 因此宜在重参考、空间滤波等混合通道信息的步骤之前完成坏道检查。


    # %% 06 插值修复 EEG 并比较波形
    # 功能与问题：保留 EEG 通道布局，用正常电极的信息估计坏电极位置的信号。
    # 实现机制：截取短片段并载入内存；根据电极坐标拟合球面样条，对坏道逐时刻估计。
    # 输入：带坏道名单和电极坐标的 raw。
    # 输出：raw_short（0～3 秒的 Raw）；eeg_data、eeg_data_interp（原始/插值 EEG Raw），
    #       二者 get_data() 均为形状 (EEG 通道数, 短片段采样点数) 的浮点 ndarray，单位 V。
    # 关键 API：Raw.copy()、Raw.crop()、Raw.load_data()、Raw.pick()、Raw.interpolate_bads()。
    # 对自己的 EEG 数据，必须先有正确的电极位置；仅有通道名称不足以完成空间插值。
    # 此示例 FIF 已包含几何信息，不应强行为这些编号套用不匹配的标准 montage。

    # 先裁剪再加载；crop 默认包含末端采样点。
    raw_short = raw.copy().crop(tmin=0.0, tmax=3.0).load_data()
    eeg_data = raw_short.copy().pick("eeg", exclude=[])  # 必须保留坏道位置，才能对这些位置进行插值。
    eeg_data_interp = eeg_data.copy().interpolate_bads(
        reset_bads=False,  # 插值后保留坏道标签，便于用红色对照；默认 True 会移除已修复通道的标签。
        method={"eeg": "spline"},  # 显式采用 EEG 默认的球面样条；不是沿时间轴补缺失采样点。
        origin="auto",  # 根据头部数字化点拟合球心；若显式提供坐标，须用头坐标系、单位米。
    )
    print("EEG interpolation shape:", eeg_data_interp.get_data().shape)
    print("EEG labels retained for comparison:", eeg_data_interp.info["bads"])

    if show_plots:
        with mne.viz.use_browser_backend("matplotlib"):
            for title, data in (
                ("EEG: before interpolation", eeg_data),
                ("EEG: after interpolation", eeg_data_interp),
            ):
                fig = data.copy().plot(  # 对绘图再取副本，避免点击图时改变待分析对象的坏道标签。
                    duration=3.0,
                    butterfly=True,  # 同类型通道叠加显示，便于看红色坏道相对通道群的位置。
                    color="#00000022",  # 八位颜色为 RRGGBBAA；末尾 22 是透明度，让叠加波形更清楚。
                    bad_color="r",
                    theme="light",
                    scalings={"eeg": 20e-6},  # 两张图固定同一尺度，20e-6 V = 20 μV；不修改实际信号。
                    proj=False,
                    remove_dc=False,  # 不仅为显示而减去窗口均值，便于直接比较原始与插值结果。
                    show=False,
                )
                # Matplotlib 浏览器 Figure 支持此方法；不要写成 layout_adjust。
                fig.subplots_adjust(top=0.88)
                fig.suptitle(title, fontsize=14)

    # reset_bads=False 只影响标签，不阻止信号被插值；后续 exclude='bads' 仍会排除这些通道。
    # 正式流程若要重新使用修复后的通道，核查结果后采用 reset_bads=True；另存修复前名单供追踪。
    # 波形“看起来正常”只能作为质控线索；它不证明未知的真实信号已被准确恢复。


    # %% 07 插值修复 MEG 梯度计并比较波形
    # 功能与问题：在保留梯度计通道布局的同时，估计异常传感器应有的测量信号。
    # 实现机制：利用正常梯度计的信号及线圈位置、方向构建场映射，估计坏通道。
    # 输入：raw_short，含设备线圈几何、坐标变换及坏道名单。
    # 输出：grad_data、grad_data_interp（原始/插值梯度计 Raw），
    #       get_data() 均为形状 (梯度计通道数, 短片段采样点数) 的浮点 ndarray，单位 T/m。
    # 关键 API：Raw.pick('grad')、Raw.interpolate_bads(method={'meg': 'MNE'})、Raw.plot()。
    # 本块对应原教程的 MEG 补充示例；只有 EEG 的真实数据不需要执行该块。

    grad_data = raw_short.copy().pick("grad", exclude=[])  # grad 只选择梯度计；mag 才是磁强计。
    grad_data_interp = grad_data.copy().interpolate_bads(
        reset_bads=False,
        method={"meg": "MNE"},  # 插值方法字典使用 meg 键，同时适用于该示例选出的 grad 通道。
        mode="accurate",  # 使用较精细的勒让德展开；这是默认值，不代表恢复真实信号的保证。
        origin="auto",
    )
    print("Gradiometer interpolation shape:", grad_data_interp.get_data().shape)
    print("Gradiometer labels retained for comparison:", grad_data_interp.info["bads"])

    if show_plots:
        with mne.viz.use_browser_backend("matplotlib"):
            for title, data in (
                ("Gradiometers: before interpolation", grad_data),
                ("Gradiometers: after interpolation", grad_data_interp),
            ):
                fig = data.copy().plot(
                    duration=3.0,
                    butterfly=True,
                    color="#00000009",  # 梯度计较多，用比 EEG 图更低的透明度减少遮挡。
                    bad_color="r",
                    theme="light",
                    scalings={"grad": 4e-11},  # 两图保持相同显示尺度；内部单位 T/m。
                    proj=False,
                    remove_dc=False,
                    show=False,
                )
                fig.subplots_adjust(top=0.88)
                fig.suptitle(title, fontsize=14)
        plt.show(block=True)  # 在桌面脚本中显示全部图，关闭窗口后脚本结束。


if __name__ == "__main__":
    main()


# %% 迁移任务：临时隐藏一条正常 EEG 通道，定量检验插值效果
#
# 【学习目的】
# 原教程处理的是真坏道，缺少它本应测得的真实波形，难以量化修复误差。
# 本任务把一条经检查正常的 EEG 通道临时标坏，保留其原始波形作比较标准。
# 保留“标记—选道—插值—比较”的主体流程，只改变目标通道和评价方式。
# 由此检验：空间插值能重建多少信息，以及保留/清除坏道标签如何影响后续选道。
#
# 【输入和输出】
# 输入：从完整 raw 取出的一个较长 EEG 片段，例如 0～30 秒；须保留电极位置。
# 输出：目标通道的原始/重建波形图，相关系数 r，RMSE（μV），以及修复前后名单。
# 原始记录仍含生理噪声，这里的比较标准是“保留下来的实测信号”，不是无噪声脑源真值。
#
# 【操作思路】
# 1. 从 raw.copy() 出发，crop(tmin=0, tmax=30).pick('eeg', exclude=[]).load_data()。
#    不从仅含 3 秒的 raw_short 或已经插值的 eeg_data_interp 开始。
#    原有坏道仍留在名单中，防止它们成为重建目标通道的供体。
#
# 2. 用 mne.pick_types(work.info, meg=False, eeg=True, exclude='bads') 找正常候选。
#    经波形检查后选一个 target（通道名字符串）。
#    用 truth = work.get_data(picks=[target])[0].copy() 留存原始波形。
#    get_data 返回 (1, n_times)，[0] 将其变为 (n_times,)；不要删除这个通道。
#
# 3. 对 work 的副本 masked，将 target 追加到 masked.info['bads']。
#    此时波形没有改变，但插值算法将不再把 target 当作正常输入。
#    用 masked.copy().interpolate_bads(reset_bads=False, method={'eeg': 'spline'})
#    得到 repaired，随后取 estimate = repaired.get_data(picks=[target])[0]。
#    标坏足以把目标信号排除出供体，不必先将其置零或加噪。
#
# 4. 画 truth 和 estimate；并计算：
#    r = np.corrcoef(truth, estimate)[0, 1]
#    rmse_uv = np.sqrt(np.mean((estimate - truth) ** 2)) * 1e6
#    r 检查波形是否同向变化；RMSE 同时反映幅值和偏移误差。
#    若某条波形标准差为零，r 无定义，应报告该情况，不能解释成 r=0。
#    不设武断的“达标数值”，也不预设插值结果必然很好。
#
# 5. 从同一个 masked 另取副本，用 reset_bads=True 再插值一次，称为 repaired_reset。
#    比较两份插值结果的数值，再分别调用 mne.pick_types(..., eeg=True)。
#    应观察到：reset_bads 只控制修复后的标签；默认选道是否纳入 target 因此改变。
#    保留 raw 中的原始坏道名单与本次人为加入的 target，便于区分处理历史。
#
# 【完成标准】
# - 原始 work 与 truth 未被修改，插值前后通道名称、顺序和时间轴一致。
# - 提供一张目标通道对照图及 r、RMSE，不用“看起来更平滑”代替误差评价。
# - 解释 reset_bads=False 时，已插值通道为什么仍被默认 mne.pick_types 排除。
# - 解释为何恢复相同通道数量，不等于恢复相同数量的独立测量信息。
#
# 【关键 API】
# Raw.copy / crop / pick / load_data / get_data / interpolate_bads；
# raw.info['bads']；mne.pick_types；np.corrcoef；np.mean；np.sqrt；plt.plot。
#
# 【参数查阅】
# https://mne.tools/stable/generated/mne.io.Raw.html
# https://mne.tools/stable/generated/mne.pick_types.html
# https://mne.tools/stable/generated/mne.Epochs.html
# https://mne.tools/stable/generated/mne.find_events.html
# https://mne.tools/stable/generated/mne.viz.plot_raw.html
