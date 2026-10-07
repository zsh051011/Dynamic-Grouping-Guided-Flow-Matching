# Dynamic Grouping-Guided Flow Matching (DyGFlow)

这是 MAC-Flow 的独立改进副本，原目录 `../mac-flow` 未被修改。

## 运行环境

RTX 5080 训练使用独立环境 `macflow_gpu_grouped`。原来的 `macflow_gpu`
仍保持不变。新环境中的 `jax`、`jaxlib`、`jax-cuda12-plugin` 和
`jax-cuda12-pjrt` 均为 0.6.2。

机器的系统工具链为 CUDA 12.8、cuDNN 9.10，驱动为 570 系列。环境中一度同时存在
pip CUDA 12.9 动态库，混载时 cuSPARSE 会因加载到 CUDA 12.8 的 `libnvJitLink`
而缺少 `__nvJitLink..._12_9` 符号；即使强制只加载 pip CUDA 12.9，持续训练也会
触发系统级 GPU reset。因此 `macflow_gpu_grouped` 已移除 pip CUDA 12.9 库，
启动时必须显式使用本机 CUDA 12.8：

```bash
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu
PATH=/usr/local/cuda-12.8/bin:$PATH
```

这样 CUDA 数学库、cuDNN、`libnvJitLink` 和 `ptxas` 属于同一套兼容的
CUDA 12.8 运行时。

该环境已在 RTX 5080 上通过撤销 CPU/microbatch 绕行后的 baseline 对齐 GPU
smoke test：`25m / Good / batch_size=16 / sequence_length=20` 的完整整批更新成功。

## 已实现

- `GroupMixer`：用全局状态生成每个智能体的重要性和组级权重，以单调方式混合个体 Q。
- 动态分组：初始单组；每 `group_update_freq` 步，使用区间内所有 batch 的平均重要性执行 select-and-kick-out。
- 分层 flow actor：组内智能体执行密集 attention；同组 token 经 max pooling 形成组 token；每个组只关联 top-k 其他组。
- SD loss：同组策略 logits 趋近、异组正余弦相似度受抑制。
- 训练和目标 GroupMixer 同步软更新；动态组容量固定为智能体数，拆组不会改变参数树形状。
- 分组感知推理：训练、flow ODE 积分和一步策略都读取同一分组；checkpoint 同时保存参数与 group ids。
- 集中式/去中心化开关：集中式使用分层 attention；去中心化关闭跨智能体特征读取，保留可学习组身份嵌入。
- 原生 SMAC NPY reader：内存映射、episode 边界安全采样、完整形状与 `filled` 校验、terminal/truncation 合并。

文档中针对 MADIFF U-Net 和扩散采样的描述不能原样移植到 MAC-Flow。本实现将 U-Net
分层注意力替换为 flow vector field 前的层次化 agent/group attention，将扩散采样替换为
多步 flow ODE 积分和蒸馏后一步 flow policy 的一致分组条件。

## 25m 训练示例

```bash
cd /media/msi/disk/ZJK/多智能体/mac-flow-grouped
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu \
PATH=/usr/local/cuda-12.8/bin:$PATH \
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n macflow_gpu_grouped \
  python smac_main.py \
  --env=smac_v1 \
  --scenario=25m \
  --agent_name=dygflow \
  --native_data_dir=../mac-flow/data/native_smac/25m/Good \
  --offline_steps=400000 \
  --batch_size=16 \
  --sequence_length=20 \
  --agent.sd_alpha=0.05 \
  --agent.change_group_value=0.8 \
  --agent.group_update_freq=10000 \
  --agent.group_attention_dim=64 \
  --agent.inter_group_top_k=2 \
  --agent.decentralized_execution=false
```

建议先用 `--offline_steps=2 --eval_interval=0 --save_interval=2` 做环境级 smoke test。
正式对比时至少固定数据、seed、训练步数、batch size、评估回合数，并同时跑原 MAC-Flow。
训练目录中的 `group_history.jsonl` 保存每次调整前使用的重要性及调整后的分组结构。

## 与原版对比边界

DyGFlow 保留原 MAC-Flow 的 Q-guidance `argmax` 和 LSTM encoder 调用方式，没有把额外
基础修复混入创新实验。结构变化只来自 GroupMixer、分层注意力、SD loss、动态分组及
分组感知推理。

## 具体代码修改

### 1. 独立工程和入口

原工程保留在 `../mac-flow`，所有创新代码位于独立目录 `../mac-flow-grouped`。

`smac_main.py` 的默认 agent 从原来的 `MACFlowDiscreteAgent` 改为
`DyGFlowDiscreteAgent`，配置文件改为 `agents/dygflow.py`。读取样本后，
程序根据 `example["infos"]["state"].shape[-1]` 自动设置 `state_dim`。

模型参数使用入口传入的完整 shape example 初始化，与原版入口行为一致。

训练循环额外维护：

```text
importance_sum
importance_count
group_history.jsonl
```

每个训练 batch 返回智能体重要性。到达 `group_update_freq` 时，用这段区间内的重要性均值
更新分组，而不是使用单个 batch。日志额外记录：

```text
train/groups/count
train/groups/min_size
train/groups/max_size
```

每个智能体/组的超网络参数仍然彼此独立，但实现为带 head 维度的参数张量，并用 `einsum`
向量化计算。它与展开 25 个 Python MLP 数学等价，同时显著缩小 JAX/XLA 编译图。

训练恢复为与 baseline 相同的整批执行方式：`batch_size=16` 的总损失在一个 JIT 中完成
前向、反向和一次 Adam 更新，不使用 microbatch。sequence LSTM、10 步 teacher flow 和
所有训练模块均在 GPU 上运行；teacher flow 使用原生 `lax.scan`，组间稀疏注意力使用
原生 `jax.lax.top_k`。此前用于排查段错误的 CPU/microbatch 绕行均已撤销。

### 2. GroupMixer 的两层混合

代码位置：`utils/networks.py::GroupMixer`。

原离散 MAC-Flow critic 使用 VDN 式求和：

```text
Q_tot = sum_i Q_i
```

改进版本首先为每个智能体建立独立超网络：

```text
w1_i = softplus(hyper_w1_i(state))
b1_i = hyper_b1_i(state)
```

然后根据当前 `group_ids` 把智能体贡献聚合为组隐状态：

```text
h_g = ELU(mean_{i in g}(Q_i * w1_i + b1_i))
```

每个潜在组槽都有独立二层超网络：

```text
w2_g = softplus(hyper_w2_g(state))
b2_g = hyper_b2_g(state)
Q_g  = |g| * mean(h_g * w2_g) + b2_g
Q_tot = sum_{active g} Q_g
```

`softplus` 保证混合权重为正，从而保持个体 Q 到联合 Q 的单调性。组隐状态先按组大小归一化，
再在输出处恢复规模，避免 25 个智能体的超网络偏置直接累加导致数值过大。全局 state 在进入
超网络前使用 LayerNorm。

训练网络和 `target_group_mixer` 使用相同 `group_ids`；target 参数按照原 MAC-Flow 的
`tau` 做软更新。

### 3. 分层 group-aware flow actor

代码位置：`utils/networks.py::GroupAwareActorVectorField`。

原 `ActorVectorField` 对每个智能体独立执行 MLP。改进版本先把 observation/LSTM feature
投影为 agent token，然后执行三层处理。

第一层是组内密集注意力：

```text
mask[i,j] = group_ids[i] == group_ids[j]
A_intra = softmax(QK^T / sqrt(d), mask)
H_intra = A_intra V
```

第二层用 max pooling 构造固定容量的组 token：

```text
H_group[g] = max_{i in g} H_agent[i]
```

第三层计算组间注意力，但每个组只保留相似度最高的 `inter_group_top_k` 个其他活跃组：

```text
A_inter = top-k-softmax(Q_group K_group^T)
H_inter = A_inter V_group
```

最后把以下特征拼接并送入原来的策略 MLP：

```text
local observation
flow action/noise
within-group context
inter-group context
flow time embedding（BC flow 路径）
```

同一个模块同时用于：

- flow matching BC velocity 预测；
- 多步 flow ODE 积分；
- 多步到一步策略蒸馏；
- evaluation 在线动作生成。

因此训练和推理使用的是同一分组结构。

### 4. 集中式和去中心化执行

配置项：

```text
--agent.decentralized_execution=false
```

集中式模式下启用组内与组间上下文，适合验证完整分层协作结构。

去中心化模式下：

- 组内 attention mask 退化为单位矩阵；
- 组内和组间上下文输出置零；
- 每个智能体仍根据 `group_ids` 使用可学习的组身份 embedding。

这样去中心化动作不会依赖其他智能体的即时 observation，同时不同组仍可形成差异化共享策略。

### 5. SD loss

代码位置：`agents/dygflow.py::_sd_loss`。

SD loss 作用于一步 flow policy 的 action logits。先对 logits 做 L2 归一化，再计算智能体两两
余弦相似度。

组内损失：

```text
L_within = mean_{same group, i != j}(1 - cosine(z_i, z_j))
```

组间损失：

```text
L_between = mean_{different groups}(max(cosine(z_i, z_j), 0)^2)
```

总 actor loss：

```text
L_actor =
    L_BC_flow
    + alpha * L_distill
    + L_Q
    + sd_alpha * (L_within + L_between)
```

使用 logits 而不是修改 LSTM encoder 的梯度路径，是为了保留原 baseline 的训练行为，同时
让 SD loss 能够直接优化策略。

### 6. 周期 select-and-kick-out

代码位置：`agents/dygflow.py::update_group_assignments`。

初始结构为：

```text
[0, 0, ..., 0]
```

GroupMixer 的 `w1` 在 embedding 维度求均值并归一化，作为每个智能体的重要性。对于组 `g`：

```text
threshold_g = mean_importance_g * change_group_value
```

重要性低于阈值的成员被迁移到新组。实现保证每个原组至少留下重要性最高的一个成员，并在
更新后重新把组编号压缩为连续整数，因此不会残留空组。

拆组会改变 JIT 静态 `group_ids`，第一次使用新结构时 JAX 会重新编译，这是正常行为。

### 7. 新组 warm-start

代码位置：`agents/dygflow.py::_warm_start_group_slots`。

JAX/Flax 要求参数树形状稳定，所以程序初始化时预先分配最多 `num_agents` 个组槽。未激活组
在拆组前没有训练信号，如果直接激活其随机参数，会造成联合 Q 和策略突然跳变。

因此创建新组时会从父组复制：

```text
group_hyper_w2
group_hyper_b2
BC flow actor group embedding
one-step flow actor group embedding
target GroupMixer 对应参数
```

新组从父组行为开始，随后通过自己的参数逐步分化。

### 8. checkpoint 修改

代码位置：`utils/flax_utils.py`。

Flax 的 `group_ids` 是 non-pytree 静态字段，默认不会包含在参数 state dict 中。因此保存文件
额外写入：

```python
save_dict["group_ids"] = tuple(agent.group_ids)
```

恢复 checkpoint 时，在恢复网络参数后再次恢复 `group_ids`。`eval_smac_checkpoint.py` 也已
改为构建 `DyGFlowDiscreteAgent`，保证评估使用训练结束时的真实分组。

### 9. native NPY reader 修改

代码位置：`utils/replay_buffers.py::populate_from_native_npys`。

reader 使用 mmap 打开数组，并依据 `path_lengths` 构造每个 episode 内部的合法序列起点：

```text
valid starts for episode e =
    offset_e ... offset_e + length_e - sequence_length
```

它不会把上一局末尾和下一局开头拼成一个序列。此外增加了：

- reader 返回 replay buffer 约定的 batch-major `(B, T, N, ...)`；
- agent 的 `_prepare_training_batch()` 再统一转成 loss 所需的 time-major
  `(T, B, N, ...)`，因此 `[1:]` 与 `[:-1]` 是同一条轨迹内的相邻 transition；
- agent 数量校验；
- obs/actions/rewards/legals/terminal/truncation/filled 形状校验；
- `sum(path_lengths)` 校验；
- `filled.npy` 有效性校验；
- terminal 与 truncation 的 done 兼容。

当前 25m Good、`sequence_length=20` 时有 156316 个合法序列，并阻止了 94970 个可能跨
episode 的扁平起点。

### 10. 保持不变的原版行为

为了直接对比已有 200 万步 baseline，以下行为没有修复或改变：

- Q-guidance 仍使用 `argmax -> one_hot`；
- LSTM encoder 仍按原版方式调用；
- actor/critic 主学习率保持 `3e-4`；
- discount、tau、flow steps 和网络隐藏层默认值保持原配置；
- 数据内容和 agent-id 拼接方式不变。

### 11. 已完成验证

目前已验证：

- Python 编译及 `git diff --check`；
- 真实 25m Good mmap 数据采样；
- 单组模型初始化和梯度更新；
- 强制拆为两组后再次梯度更新；
- 新组超网络与组 embedding 的父组复制；
- 集中式在线动作生成；
- 去中心化在线动作生成；
- 25 个智能体合法动作 mask；
- checkpoint 参数和 group ids 保存/恢复；
- `smac_main.py` 完整 1-step 训练入口。
- GPU 新环境已完成 baseline 对齐模式的正式 `batch=16, sequence=20, agents=25`
  单步更新。

这些测试确认代码路径可运行，但不等价于完整性能实验。正式 200 万步前仍建议先跑
10 万至 40 万步，观察 Q、梯度、group loss、分组数量和 evaluation return 是否稳定。

## GPU 编译故障恢复

如果日志出现 `backend_compile` 内的 `Segmentation fault`，随后 `jax.devices()` 只返回
`CpuDevice`，说明 CUDA backend 在编译器崩溃后没有恢复。先重启机器，再验证：

```bash
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu \
PATH=/usr/local/cuda-12.8/bin:$PATH \
conda run -n macflow_gpu_grouped python -c "import jax; print(jax.devices())"
```

必须看到 `CudaDevice(id=0)` 才启动正式训练。超网络仍使用数学等价的向量化独立参数，
避免为 25 个智能体展开重复的 Python 模块。

## 读取方式审查

SMAC 读取路径的核心设计是合理的：不复制大数组，依据 `path_lengths` 枚举合法起点，
不会让长度为 `sequence_length` 的片段跨 episode。当前 25m Good 数据已核验为：

- observations: `(251305, 25, 250)`
- states: `(251305, 950)`
- actions: `(251305, 25)`
- legals: `(251305, 25, 31)`
- episodes: `5000`，且 `sum(path_lengths) == 251305`

原修改版的主要风险是未读取 `filled.npy`、只校验部分数组、训练时没有将 truncation 合并到
done。本副本已修正。MA-MuJoCo 的原生读取函数仍使用整批写入 Flashbax，对于大数据集会有
明显峰值内存；如实验转到 MA-MuJoCo，应再改成与 SMAC 相同的 mmap 索引采样器。

## 动态分组运行时修复

正式训练在动态分组时曾出现：

```text
_warm_start_group_slots -> Array.at -> XlaRuntimeError:
INTERNAL: Failed to launch ptxas
```

原因是组参数 warm-start 使用 GPU `Array.at`，会在训练中途启动一次额外的 PTX 编译；
同时 `group_ids` 原来是 JIT 静态字段，分组值变化后还会触发整个训练更新函数重新编译。
当前副本改为：

- 在主机端逐值复制父组参数，再放回原设备；
- 将固定形状的 `group_ids` 保存为动态 `int32` JAX 数组，使不同分组值复用同一计算图。

这两项只改变参数复制和 JIT 缓存方式，不改变分组规则、复制后的数值、loss 或梯度公式。
已用两步强制拆组测试覆盖“更新、拆组、再次更新”路径。

运行正式实验前还需保证：

- `nvidia-smi` 可正常返回；
- 数据盘留有足够空间。2026-07-29 故障时该盘仅余约 653 MB，无法可靠保存编译缓存和
  100k/200k 等 checkpoint。
