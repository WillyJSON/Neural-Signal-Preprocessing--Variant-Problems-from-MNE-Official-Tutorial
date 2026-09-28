"""MNE EEG 重参考教程的注释参考版。

输入：MNE sample 前 60 秒的 EEG 041～059。
处理：演示补回参考通道、单电极、平均参考、REST 和双极导联。
输出：按需绘制参考前后的波形，保留原教程说明及迁移任务。
依赖：MNE、Matplotlib。
"""

# -*- coding: utf-8 -*-
# MNE-Python：设置和更改 EEG 参考（按数据变换分块的中文注释版）
#
# 功能与问题
# EEG 保存的是电位差：x_i(t) = 第 i 个电极的电位 - 采集参考电极的电位。
# 因此，采集参考中的波动会进入每个通道。本文件演示如何补回未保存的参考通道，
# 并把数据转换为单电极参考、平均参考、REST 无穷远参考和双极导联，便于比较其效果。
#
# 实现机制
# 常规重参考在每个时刻从各有效 EEG 通道减去同一个参考值 r(t)：y_i(t)=x_i(t)-r(t)。
# 单电极参考的 r(t) 来自指定通道；平均参考的 r(t) 来自有效 EEG 通道的空间平均。
# 投影器保存平均参考的线性变换，允许先预览、后应用；REST 借助头模型和前向模型估计
# 无穷远参考；双极导联则把指定的两个通道相减，生成一个新的虚拟通道。
# 普通重参考保留有效通道间的电位差，不能恢复各电极的绝对电位，也不保证自动去除伪迹。
#
# 板块架构：每块以得到一种完整的数据、元数据或模型结果为边界
# 1. 准备 EEG 演示数据：读取、裁剪并选择后续分析使用的 EEG 通道。
# 2. 补回缺失的原始参考电极：为未保存的旧参考创建全零通道。
# 3. 切换到指定电极参考：以 EEG 050 为新的单电极参考。
# 4. 应用平均参考：以所有有效 EEG 通道的瞬时平均值作为参考。
# 5. 创建可开关的平均参考：将平均参考作为投影器保存并预览其效果。
# 6. 建立 REST 重参考模型：构建头模型、源空间和 EEG 前向模型。
# 7. 应用 REST 无穷远参考：利用前向模型估计无穷远参考下的 EEG。
# 8. 创建双极导联：计算 EEG 054 与 EEG 055 之间的电位差。
# 其中 2→3 是连续处理；4、5、7、8 是从 raw 出发的不同示例分支。
# 第 5 块只给 raw 增加投影器元数据，第 6 块删除该未应用投影器；raw 的波形保持原参考。
#
# 数据约定
# Raw 是 MNE 的连续记录容器，包含波形、采样率、通道信息、坏通道标记和投影器等。
# raw.get_data() 返回二维 NumPy 数组，形状为 (通道数, 时间点数)，本例 EEG 数值单位为 V。
# raw.times 是形状为 (时间点数,) 的秒数数组；raw.info 是字典式的测量元数据对象。
# 文中的 N 表示裁剪后的时间点数；官方示例 N=36038、采样率约 600.61 Hz。
# 每次重参考都保持 N 和采样率不变。图中的 μV 显示不会改变内存数据的 V 单位。
#
# 运行与来源
# 安装本文件核对的版本：python -m pip install "mne==1.13.2"
# 运行：python 55_setting_eeg_reference_annotated.py
# 首次调用 sample.data_path() 可能下载整个 sample 数据集；裁剪 60 秒不会减少下载体积。
# 原有板块注释保留用于阅读；调用 main() 执行流程，show_plots 控制是否绘图。
# Matplotlib 后端设置和末尾 plt.show() 支持独立脚本；主要数值计算沿用教程。
# 19 个通道的选择用于方便教学绘图，真实分析应根据电极布局确定参与参考的通道集合。
# 教程：https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html
# API：https://mne.tools/stable/generated/mne.set_eeg_reference.html
# API：https://mne.tools/stable/generated/mne.add_reference_channels.html
# API：https://mne.tools/stable/generated/mne.set_bipolar_reference.html
# Authors: The MNE-Python contributors.
# License: BSD-3-Clause（原代码的完整许可声明保留在文件末尾）
# Copyright the MNE-Python contributors.
# 中文说明及独立运行调整：本文件新增。


# %% 板块 1：准备 EEG 演示数据
# 功能与问题：取得后续所有参考方法的共同输入，减少演示时的内存占用和通道显示数量。
# 实现机制：先延迟读取 FIF 的结构信息，裁剪到起始 60 秒，再加载波形并按名称选择 EEG。
# 输入：sample 数据集中的 sample_audvis_raw.fif，包含多种通道的连续记录及其测量信息。
# 输出：raw，已预加载的 mne.io.Raw 对象；波形形状 (19, N)，通道为 EEG 041～EEG 059。
#       原有坏通道 EEG 053 仍在对象中；raw.info['bads'] 保存其标记，选中不等于质量合格。
#       raw.plot() 生成原参考下的波形浏览图，供后续比较。
# 关键 API：sample.data_path() 定位数据；read_raw_fif() 读取；crop() 裁剪；
#           load_data() 载入内存；pick() 选择通道；plot() 可视化。


from pathlib import Path

import matplotlib.pyplot as plt  # 用于在独立运行 .py 文件时保持图窗显示。
import mne


def main(show_plots: bool = True, path: Path | None = None) -> None:
    if show_plots:
        # 选择 Raw 波形浏览器后端，使返回图支持后文的 Matplotlib 排版方法。
        mne.viz.set_browser_backend("matplotlib")
    if path is None:
        path = Path(mne.datasets.sample.data_path()) / "MEG" / "sample"
        path = path / "sample_audvis_raw.fif"
    sample_data_raw_file = path
    # preload 默认 False，尚未把全部波形载入内存；verbose=False 仅抑制本次调用日志。
    raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
    raw.crop(tmax=60).load_data()  # 两个方法均原地修改并返回自身；默认从 0 秒裁剪且包含终点，先裁剪再加载可减少内存使用。
    # range 不含 60；:02 把 n 格式化为至少两位并补零，再与前面的固定 0 拼成 EEG 041 等名称。
    raw.pick([f"EEG 0{n:02}" for n in range(41, 60)])
    if show_plots:
        raw.plot()  # 波形图默认只显示一个时间窗口；未显示全部 60 秒不代表数据被再次裁剪。

    # crop 的时间相对 Raw 当前起点；它保留样本索引信息，不能把 raw.first_samp 当作相对秒数。
    # pick() 原地改变通道集合；这里显式指定名称，因此也保留被标记为坏的 EEG 053。
    # 重参考会修改波形，要求数据已载入内存；只有 read_raw_fif() 而没有 load_data() 会出错。


    # %% 板块 2：补回缺失的原始参考电极
    # 功能与问题：原参考电极若未作为通道保存，直接换参考就无法在结果中查看该电极的波形。
    # 实现机制：在旧参考坐标下，旧参考对自身的电位差恒为 0，因此增加一行全零数据。
    #           这一步只是补回可表示的通道；完成下一块的重参考后，该通道才会出现非零波形。
    # 输入：raw，波形形状 (19, N) 的已预加载 Raw。
    # 输出：raw_new_ref，独立的 Raw 副本，波形形状 (20, N)，末尾新增 EEG 999；raw 保持不变。
    #       新通道的零值表示“相对于旧参考的电位差为零”，不能解释为该位置没有神经活动。
    # 关键 API：mne.add_reference_channels() 添加全零 EEG 参考通道；Raw.plot() 显示结果。

    # copy 默认 True；输入须预加载，通道名不能与现有名称重复。
    raw_new_ref = mne.add_reference_channels(raw, ref_channels="EEG 999")
    if show_plots:
        raw_new_ref.plot()  # EEG 999 此时是平线；如未出现在初始可见范围内，可在波形浏览器中滚动查看。

    # EEG 999 是教程的演示名称，不能据此认定实际采集时的参考电极就叫这个名字。
    # 迁移到真实数据时，应按采集记录确认缺失的参考电极；不要给任意缺失电极补零。
    # 添加参考后仍需核对电极坐标；若用 set_montage() 设置布局，应在添加真实参考通道后设置。


    # %% 板块 3：切换到指定电极参考
    # 功能与问题：把同一组连续波形从旧参考转换到一个指定的、已有的 EEG 电极参考。
    # 实现机制：逐时刻用每个非坏 EEG 通道减去变换前 EEG 050 的波形。
    #           EEG 050 减自身后为零；补入的 EEG 999 由 0 变成 -x_050(t)，表示旧参考相对新参考。
    # 输入：raw_new_ref，含全零 EEG 999 的 Raw，波形形状 (20, N)。
    # 输出：同一个 raw_new_ref 对象，形状仍为 (20, N)；其有效 EEG 波形已完成重参考。
    #       EEG 053 保留原值，因为它已被标记为坏通道；raw 仍保持原参考。
    # 关键 API：Raw.set_eeg_reference(ref_channels=...) 原地更改参考并返回 Raw 自身。

    # projection 默认 False，立即改写波形；字符串必须与 ch_names 中的完整通道名一致。
    raw_new_ref.set_eeg_reference(ref_channels="EEG 050")
    if show_plots:
        raw_new_ref.plot()  # 检查 EEG 050 变平、EEG 999 出现波形；EEG 053 未被重参考，不能按同一参考直接解读。

    # 以下保留教程的其他电极示意；sample 中没有这些通道，因此保持注释状态。
    # raw.set_eeg_reference(ref_channels="A1")  # 单个已有电极作参考；执行时会原地修改 raw。
    # raw.set_eeg_reference(ref_channels=["M1", "M2"])  # 两个参考电极先逐时刻求平均，再从有效 EEG 通道中减去。
    # 显式传入参考名称前要检查其质量：不能把“坏通道不会被更新”误解为“显式指定的坏参考会自动忽略”。
    # 此处用的是 Raw 方法，返回 Raw；顶层 mne.set_eeg_reference() 返回 (inst, ref_data) 元组。


    # %% 板块 4：应用平均参考
    # 功能与问题：用一组有效 EEG 电极的平均信号定义参考，减少对某一个参考电极的依赖。
    # 实现机制：每个时刻分别计算 18 个非坏 EEG 通道的均值，再从这 18 个通道中减去该均值。
    #           平均沿“通道轴”进行，不是对每个通道沿时间轴去均值，也不是时间基线校正。
    # 输入：原始分支 raw，形状 (19, N)；本块不使用已换成 EEG 050 参考的 raw_new_ref。
    # 输出：raw_avg_ref，形状 (19, N) 的独立 Raw；非坏 EEG 通道逐时刻均值约为零。
    #       EEG 053 既不参加均值计算，也不被修改，因此把它包含在内时，19 通道总均值未必为零。
    # 关键 API：Raw.copy() 复制波形与元数据；set_eeg_reference('average') 立即执行平均参考。

    # 必须先 copy() 才能保留 raw；给原地方法的返回值换变量名并不会自动复制。
    raw_avg_ref = raw.copy().set_eeg_reference(ref_channels="average")
    if show_plots:
        raw_avg_ref.plot()  # 显示已写入 raw_avg_ref 的平均参考波形；此时关闭绘图投影也无法恢复其原参考数据。

    # 这里沿用教程的 19 通道演示输入。若真实分析希望将未保存的原参考电极也计入平均，
    # 应在“尚未更改参考”的数据上先补回该全零通道，再做平均参考；若已记录的有效通道数为 K，
    # 加上该参考后，平均分母应为 K+1。
    # 只对选出的部分电极平均，得到的是这部分电极的平均参考，不能当作完整电极阵列的平均参考。


    # %% 板块 5：创建可开关的平均参考
    # 功能与问题：暂缓改写波形，同时允许比较平均参考的效果，并为后续通道选择保留调整余地。
    # 实现机制：在 info['projs'] 中加入平均参考的 Projection 对象；它存放构建线性投影的向量。
    #           对 g 个有效 EEG 通道，其作用等价于每个时刻减去这 g 个通道的平均值。
    #           raw.plot(proj=True) 将该变换用于显示，proj=False 显示未应用它的波形。
    # 输入：raw，形状 (19, N)，保持原参考波形。
    # 输出：同一个 raw，波形数值与形状不变，info['projs'] 新增 active=False 的平均参考投影器；
    #       另输出 Original / Average 两幅 Matplotlib 浏览图。本例投影器覆盖 18 个非坏 EEG 通道。
    # 关键 API：set_eeg_reference(projection=True) 添加投影器；info['projs'] 查看状态；
    #           use_browser_backend() 临时选择后端；Raw.plot(proj=...) 控制预览。

    # projection=True 仅支持平均参考；它修改投影器元数据，当前预加载波形尚未被减去均值。
    raw.set_eeg_reference("average", projection=True)
    print(raw.info["projs"])  # projs 是 Projection 对象列表；active=False 表示尚未真正应用到记录中。

    if show_plots:
        # zip 按位置配对：Original/False 和 Average/True，而不是生成四种组合。
        for title, proj in zip(["Original", "Average"], [False, True]):
            # 上下文内使用 Matplotlib 浏览器，退出时恢复先前后端；保证 fig 支持后面的排版方法。
            with mne.viz.use_browser_backend("matplotlib"):
                # len(raw) 是通道数 19，不是时间点数；proj 参数影响显示，不改写 raw 的波形。
                fig = raw.plot(proj=proj, n_channels=len(raw))
            fig.subplots_adjust(top=0.9)  # top 是图窗中的相对位置，给总标题留空间；不改变信号幅度或时间范围。
            # f-string 把当前标题填入字符串；size、weight 仅控制标题外观。
            fig.suptitle(f"{title} reference", size="xx-large", weight="bold")

    # 若确实要得到应用后的独立结果，可使用下一行；为了继续原教程的分支，此处不执行。
    # raw_applied = raw.copy().apply_proj()  # 真正改写副本波形并激活其投影器；会应用该副本上的全部可用投影器。
    # 已经应用的投影不能靠 plot(proj=False) 或 del_proj() 撤销，因此需要原始数据时应事先留副本。
    # 在投影尚未应用时，后续有效通道集合的变化可在构建投影算子时得到反映；已做的数值减法不会自动重算。
    # 若还有影响 EEG 的未应用 SSP 投影，直接数值重参考可能被阻止；添加平均参考投影器可延后共同处理。


    # %% 板块 6：建立 REST 重参考模型
    # 功能与问题：REST 需要知道脑内电流源如何在电极上形成电位，单有波形矩阵不足以计算。
    # 实现机制：清理上一块的未应用投影器；根据数字化点拟合球形头模型，在球内建立候选源网格，
    #           再计算每个源的三个方向分量到每个 EEG 电极的映射，即 lead field（导联场矩阵）。
    # 输入：raw.info 中的电极位置、数字化头部点等测量信息；raw 自带上一块添加的未应用投影器。
    # 输出：sphere，mne.bem.ConductorModel 球形导体模型；src，mne.SourceSpaces 源空间对象；
    #       forward，mne.Forward 字典式对象，forward['sol']['data'] 是形状 (19, 3*S) 的 NumPy 矩阵，
    #       S=forward['nsource'] 为有效源位置数。raw 的投影器被删除，波形仍为原参考下的 (19, N)。
    #       S 随头部几何而变；教程本次展示 S=537，并不是所有数据都固定有 537 个源。
    # 关键 API：del_proj() 删除未应用投影器；make_sphere_model() 建头模型；
    #           setup_volume_source_space() 布置候选源；make_forward_solution() 计算前向映射。

    raw.del_proj()  # 不传索引时删除所有可删除的投影器，不仅是平均参考；不能借此撤销已经应用的投影。
    sphere = mne.make_sphere_model(
        "auto",  # 第一个位置参数 r0：根据 raw.info 的数字化点估计球心；模型内部空间坐标使用米。
        "auto",  # 第二个位置参数 head_radius：自动拟合头皮球半径；需要足够且有效的数字化点。
        raw.info,  # 第三个位置参数 info：提供拟合所需的头部几何信息，而不是把 EEG 波形传给头模型。
    )
    src = mne.setup_volume_source_space(
        sphere=sphere,  # 使用已建球模型限定源空间范围；返回源位置及其属性，不是记录到的神经活动。
        exclude=30.0,  # 单位为 mm：排除距球心 30 mm 内的源；不是排除距头皮 30 mm 内的源。
        pos=15.0,  # 浮点数时单位为 mm，表示网格间距；教程用粗网格加快计算，正文提到实际分析常用更细的 5 mm。
    )
    forward = mne.make_forward_solution(
        raw.info,  # 根据现有通道及其坐标计算电极观测；本例前向矩阵含 19 个 EEG 通道，包括坏通道的模型行。
        trans=None,  # 不提供 MRI→头坐标配准文件，使用单位变换；适用于本例的球模型设置，不能直接套用到未配准的真实 MRI。
        src=src,  # 传入候选源空间；默认自由方向，每个有效源位置有三个方向分量，所以矩阵列数为 3*S。
        bem=sphere,  # bem 参数也接受球形 ConductorModel；虽然参数名为 bem，本例没有建立真实头部的 BEM 网格。
    )

    # setup_volume_source_space() 的默认 mindist=5.0 mm 还会排除过于接近边界的候选源。
    # pos、exclude、mindist 使用 mm，而模型中保存的坐标通常使用 m；不能按同一个单位传参。
    # EEG 999 只在 raw_new_ref 分支中，本块用原始 raw 建模，不能直接把未核实位置的新增通道用于前向计算。


    # %% 板块 7：应用 REST 无穷远参考
    # 功能与问题：借助电生理前向模型，把记录转换为对“无穷远处电位为零”参考的模型估计。
    # 实现机制：先得到平均参考的波形和前向矩阵，再利用该矩阵的伪逆估计公共电位分量，
    #           将该分量加回平均参考波形。结果依赖模型与电极位置的质量，是估计而非绝对电位的实测。
    # 输入：raw，原参考下 (19, N) 的 Raw；forward，上一块构建且通道匹配的 mne.Forward。
    # 输出：raw_rest，独立 Raw，波形形状仍为 (19, N)；18 个有效 EEG 通道经过 REST，坏通道保留原值。
    #       同时生成原参考与 REST 的两幅图，采用相同显示尺度进行比较。
    # 关键 API：Raw.copy() 复制数据；set_eeg_reference('REST', forward=...) 执行变换；
    #           Raw.plot(scalings=...) 统一绘图尺度。

    # REST 必须提供 Forward；内部按有效 EEG 通道匹配模型行，不能只凭矩阵行数相同就认为通道一致。
    raw_rest = raw.copy().set_eeg_reference("REST", forward=forward)

    if show_plots:
        # _raw 是本轮所选 Raw 对象的变量名，前导下划线不表示复制或特殊数据类型。
        for title, _raw in zip(["Original", "REST (∞)"], [raw, raw_rest]):
            # 与投影预览相同，保证拿到可用 Matplotlib 方法调整的浏览图。
            with mne.viz.use_browser_backend("matplotlib"):
                fig = _raw.plot(
                    n_channels=len(raw),  # raw 与 raw_rest 都有 19 通道；这里只控制一次显示多少通道。
                    # EEG 显示尺度为 5×10⁻⁵ V，即 50 μV；这是绘图尺度，不是给数据乘 5e-5。
                    scalings=dict(eeg=5e-5),
                )
            fig.subplots_adjust(top=0.9)  # 给标题留出图窗顶部空间。
            # 标题注明本图所用参考，方便在相同尺度下比较。
            fig.suptitle(f"{title} reference", size="xx-large", weight="bold")

    # %% 板块 8：创建双极导联
    # 功能与问题：得到两个指定位置之间的电位差，用于查看这对电极的相对活动。
    # 实现机制：逐时刻计算 EEG 054 - EEG 055，共同的旧参考项在相减时抵消；
    #           将结果作为新通道添加到副本，默认删除参与相减的两个原通道，保留其余通道。
    # 输入：raw，形状 (19, N) 的原参考 Raw，含有效通道 EEG 054 和 EEG 055。
    # 输出：raw_bip_ref，形状 (18, N) 的独立 Raw，即 19-2+1 个通道；
    #       新通道默认名为 EEG 054-EEG 055，其余 17 个通道的波形保留；raw 本身不变。
    # 关键 API：mne.set_bipolar_reference() 以 anode-cathode 构建虚拟通道；
    #           默认 copy=True、drop_refs=True，返回与输入同类的数据对象。

    raw_bip_ref = mne.set_bipolar_reference(
        raw,  # 必须是已预加载的数据；默认在副本上操作，不需要额外 raw.copy()。
        anode="EEG 054",  # 被减数，计算式中的正号项；它的通道类型和大部分元数据会用于新通道。
        cathode="EEG 055",  # 减数，计算式中的负号项；调换 anode/cathode 会让所得波形整体反号。
    )
    if show_plots:
        raw_bip_ref.plot()  # 新虚拟通道通常位于通道列表末尾；这只构建了一对双极导联，并未把全部 EEG 改为双极导联。

    # 若要保留参与相减的两个原通道，给上面的函数增加 drop_refs=False；本例将得到 20 个通道。
    # 若任一输入通道被标记为坏，默认 on_bad='warn' 会警告并把生成的虚拟通道也标为坏。
    # 原教程另有 F3/F4 的示意；本数据不含这两个名称，因此保持注释，并使用当前正确的顶层 API：
    # mne.set_bipolar_reference(raw, anode="F3", cathode="F4")  # MNE 1.13.2 没有
    # raw.set_bipolar_reference() 这个方法。
    # 当前 1.13.2 API/实现默认继承阳极位置并设 coil_type=EEG_BIPOLAR；教程正文写“位置为 (0,0,0)”
    # 与当前实现不一致。继承的位置也不代表多出了一枚真实电极，后续定位分析应核对通道定义。

    if show_plots:
        plt.show()  # 独立脚本运行时统一保持图窗，通常关闭全部图窗后退出；交互式编辑器可能已在前面显示图形。


    # 教程末尾的应用边界：EEG 源重建
    # 若继续做 MNE 的 EEG 逆解，应使用平均参考投影器策略；上面的 REST 和双极示例不构成源重建流程。
    # 对合适的原始 EEG 分支，可先调用 raw.set_eeg_reference('average', projection=True)。
    # 投影器可在后续按有效通道构造参考，避免删通道后仍沿用旧集合的数值均值；相关 MNE 逆解接口
    # 会检查平均参考投影器。已立即做平均参考，并不等于对象中已经具有平均参考投影器。
    # 使用公共平均参考也可避免把某个单独参考电极的前向建模误差集中传播到全部电极。


if __name__ == "__main__":
    main()


# 原代码许可全文（BSD-3-Clause）
# 来源：https://github.com/mne-tools/mne-python/blob/v1.13.2/LICENSE.txt
# Copyright 2011-2025 MNE-Python authors
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be
# used to endorse or promote products derived from this software without specific prior
# written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

# %% 迁移任务：验证删减通道对平均参考的影响
#
# 学习目的
# 理解平均参考依赖于参与计算的电极集合；掌握在通道筛选后保持参考一致的方法。
# 进一步区分“已经改写波形的平均参考”和“尚未应用的平均参考投影器”。
# 本任务复用原教程的数据准备、通道选择、平均参考和投影操作，增加一次通道删减与数值比较。
#
# 任务设定
# 从本文件末尾仍保留原参考波形的 raw 出发，复制并排除已标记为坏的 EEG 053，
# 得到包含 18 个有效 EEG 通道的 base，数据形状为 (18, N)，单位为 V。
# 再指定删除 EEG 041、EEG 042、EEG 043，得到统一的保留通道列表 keep_names。
# 以下三个方案均从 base 的独立副本开始，最终通道名称和排列顺序必须相同。
#
# 大致实现思路
#
# 方案 A：先应用平均参考，再删除通道
# 对 18 个有效通道立即执行平均参考，然后只保留 keep_names 中的 15 个通道。
# 得到 raw_a。它保留的是按照原来 18 个通道计算的参考。
#
# 方案 B：先删除通道，再应用平均参考
# 先保留 keep_names 中的 15 个通道，再立即执行平均参考。
# 得到 raw_b。它使用的是当前 15 个通道的平均值。
#
# 方案 C：先创建投影器，删除通道后再应用
# 在包含 18 个通道的副本上添加平均参考投影器，但暂不应用。
# 随后保留 keep_names 中的 15 个通道，最后调用 apply_proj()。
# 得到 raw_c。检查投影器是否根据保留下来的通道构建了正确的平均参考变换。
#
# 三个结果均为 Raw 对象；get_data() 返回形状 (15, N) 的二维数组。
#
# 需要验证的机制
# 1. 分别计算三个结果在每个时刻的通道平均值。
#    预期：方案 A 一般不再满足零均值；方案 B、C 的均值接近零。
#    原因：删减通道不会让已经减去的参考信号自动更新。
#
# 2. 比较 raw_b 和 raw_c 的完整波形数组。
#    预期：二者在浮点误差范围内相同。
#    这说明延迟应用投影器，可以在确定最终通道集合后执行相应的平均参考。
#
# 3. 比较方案 A、B 中 EEG 044 与 EEG 045 的电位差。
#    预期：虽然单通道波形可能不同，这一对电极的差分波形仍相同。
#    原因：参考变化给各通道引入的是同一个随时间变化的公共偏移，相减时会抵消。
#    这里复用了原教程双极导联的核心思路：比较两个电极之间的电位差。
#
# 关键 API
# raw.info["bads"]                       —— 获取坏通道名称列表。
# raw.copy()                            —— 创建独立副本，避免各方案互相影响。
# raw.pick(channel_names)               —— 原地保留指定名称的通道。
# raw.set_eeg_reference("average")       —— 立即执行平均参考。
# raw.set_eeg_reference("average",
#                       projection=True) —— 添加尚未应用的平均参考投影器。
# raw.apply_proj()                      —— 真正应用对象中的投影器并改写波形。
# raw.get_data()                        —— 提取形状为 (通道数, 时间点数) 的数组。
# data.mean(axis=0)                     —— 沿通道轴求均值，返回长度为 N 的时间序列。
# np.max(np.abs(array))                 —— 把均值残差或波形差异汇总为最大绝对值。
# np.allclose(a, b, atol=1e-12, rtol=0)  —— 判断两组以 V 为单位的数据是否在容差内相同。
#
# 完成标准
# 输出三个方案的“通道均值最大绝对值”、B/C 波形是否一致、A/B 电极差分是否一致。
# 根据结果解释：为什么删通道后可能需要重新计算平均参考，以及延迟投影解决了什么问题。
# 注意检查的是每个时刻跨通道的均值；mean(axis=1) 得到的是各通道的时间均值，不适用于此验证。
