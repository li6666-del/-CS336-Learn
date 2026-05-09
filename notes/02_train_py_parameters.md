# train.py 参数说明

这份文档解释 `04_training_loop/train.py` 里的 `TrainConfig` 参数。当前默认配置的目标不是训练 GPT-3 Small，而是先在 TinyStories 上跑通一个小型 decoder-only Transformer 的完整训练闭环。

## 路径和缓存

### `data_path`

默认值：

```python
ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
```

这是原始训练文本路径。训练开始时，脚本会读取这个文本，并用 BPE tokenizer 编码成 token id。

如果以后换数据集，就改这个参数。

### `vocab_path`

默认值：

```python
ROOT / "01_bpe" / "vocab.bin"
```

这是 BPE 词表文件。`Tokenizer.from_files(...)` 会从这里加载 token id 到 bytes 的映射。

### `merges_path`

默认值：

```python
ROOT / "01_bpe" / "merges.bin"
```

这是 BPE merge 规则文件。编码新文本时，tokenizer 会按这些 merge 规则合并 byte token。

### `token_cache_path`

默认值：

```python
ROOT / "data" / "tinystories_tokens.pt"
```

第一次训练时，从文本编码 token 会比较慢，所以脚本会把编码结果缓存成 `.pt` 文件。以后再次训练时，会直接加载这个缓存，不必重复 tokenize。

### `checkpoint_dir`

默认值：

```python
ROOT / "checkpoints"
```

checkpoint 保存目录。训练过程中定期保存的模型参数和最终的 `latest.pt` 都会放在这里。

### `rebuild_token_cache`

默认值：

```python
False
```

如果为 `False`，且 `token_cache_path` 已存在，就直接加载缓存。

如果你改了原始文本、BPE 词表、merge 规则，应该加：

```powershell
--rebuild_token_cache
```

强制重新编码数据。

## 模型结构参数

### `vocab_size`

默认值：

```python
0
```

`0` 表示自动使用 tokenizer 的词表大小：

```python
len(tokenizer.vocab)
```

现在你的 BPE 训练目标是 10000，所以实际模型输出维度一般是 `10000`。

不要随便手动改小它，否则 target token id 可能超过模型输出维度，导致训练报错。

### `context_length`

默认值：

```python
256
```

它就是训练 batch 里的 `T`，表示模型一次最多看多少个 token。

训练时：

```python
x.shape == (batch_size, context_length)
y.shape == (batch_size, context_length)
```

进入 embedding 后：

```python
x.shape == (batch_size, context_length, d_model)
```

默认用 `256` 是为了先跑通训练，显存压力较小。以后可以逐步试：

```text
256 -> 512 -> 1024 -> 2048
```

注意 attention 计算量大致随 `context_length^2` 增长，所以把 256 提到 1024，attention 部分会明显变慢、占更多显存。

### `d_model`

默认值：

```python
384
```

这是每个 token 的隐藏向量维度，也可以理解成 Transformer 内部通道数 `C`。

更大的 `d_model` 会提升模型容量，但也会显著增加参数量和显存消耗。

参考级别：

```text
调试小模型: 128 / 256
当前小模型: 384
中等练习:   512
GPT-3 Small: 768
```

### `num_layers`

默认值：

```python
6
```

Transformer block 的层数。层数越多，模型越深，表达能力越强，但训练更慢。

参考级别：

```text
调试:       2
当前默认:   6
中等练习:   8
GPT-3 Small: 12
```

### `num_heads`

默认值：

```python
6
```

self-attention 的 head 数量。

必须满足：

```python
d_model % num_heads == 0
```

因为每个 head 的维度是：

```python
head_dim = d_model // num_heads
```

当前默认：

```text
d_model = 384
num_heads = 6
head_dim = 64
```

`head_dim = 64` 是很常见的选择。GPT-3 Small 也是 `d_model=768, num_heads=12, head_dim=64`。

### `d_ff`

默认值：

```python
1536
```

这是前馈网络中间层维度。当前是：

```python
d_ff = 4 * d_model
```

即：

```text
1536 = 4 * 384
```

这是经典 Transformer 的常见比例。你的模型里用的是 SwiGLU FFN，未来也可以尝试略小一点的比例，但现在保持 `4 * d_model` 最简单稳妥。

### `rope_theta`

默认值：

```python
10000.0
```

这是 RoPE 位置编码的频率基数。短上下文训练时，`10000` 是常用默认值。

当你将 `context_length` 扩到很长，比如 4096、8192 以上时，才需要更认真考虑调整它。

### `rms_norm_eps`

默认值：

```python
1e-5
```

RMSNorm 里的数值稳定项，防止除以非常小的数。

通常不用调。

## 训练控制参数

### `batch_size`

默认值：

```python
16
```

这里的 batch size 是一次取多少条序列。每一步实际训练 token 数是：

```text
batch_size * context_length
```

当前默认：

```text
16 * 256 = 4096 tokens / step
```

显存不够时，优先降低 `batch_size`。显存充足时，可以增大它，让梯度估计更稳定。

### `max_steps`

默认值：

```python
5000
```

总训练步数。每一步会采样一个 batch，前向、计算 loss、反向传播、更新参数。

当前默认适合先观察 loss 是否下降。正式训练时通常需要更多 step。

### `eval_interval`

默认值：

```python
250
```

每隔多少 step 在 train 和 valid 上估计一次 loss。

太小会频繁打断训练；太大又不容易观察训练状态。`250` 是一个适中的默认值。

### `eval_iters`

默认值：

```python
20
```

每次 eval 时采样多少个 batch 来估计平均 loss。

值越大，评估越稳定，但耗时越长。调试时可以设成：

```powershell
--eval_iters 2
```

正式观察曲线时可以设为 `20`、`50` 或更高。

### `checkpoint_interval`

默认值：

```python
1000
```

每隔多少 step 保存一次 checkpoint。

如果设置为 `0`，则训练过程中不保存中间 checkpoint，只在最后保存 `latest.pt`。

调试时常用：

```powershell
--checkpoint_interval 0
```

### `train_split`

默认值：

```python
0.9
```

表示 90% token 用于训练，10% token 用于验证。

valid loss 可以帮助判断模型是不是只是在记忆训练集。

### `grad_clip`

默认值：

```python
1.0
```

梯度裁剪阈值。训练时如果梯度过大，可能导致 loss 爆炸。`1.0` 是训练 Transformer 时常用的稳妥选择。

如果设置为 `0` 或负数，脚本会跳过梯度裁剪。

### `seed`

默认值：

```python
1337
```

随机种子，用来让随机初始化和 batch 采样更可复现。

## 优化器和学习率参数

### `learning_rate`

默认值：

```python
3e-4
```

这是 AdamW 的最大学习率，也传给 cosine schedule 的 `max_lr`。

对于当前 6 层、384 维的小模型，`3e-4` 是比较保守的起点。以后如果训练稳定，可以尝试：

```text
3e-4 -> 5e-4 -> 6e-4
```

如果 loss 出现明显爆炸或 NaN，优先降低学习率。

### `min_lr`

默认值：

```python
3e-5
```

cosine decay 最后下降到的最低学习率。

当前设置为：

```text
min_lr = learning_rate / 10
```

这是常见做法。

### `warmup_steps`

默认值：

```python
200
```

训练开始时，学习率从较小值逐步升到 `learning_rate`，这个阶段叫 warmup。

Transformer 训练初期比较脆弱，warmup 可以减少刚开始梯度不稳定的问题。

当前默认 `max_steps=5000`，所以 `warmup_steps=200` 大约占 4%。

### `weight_decay`

默认值：

```python
0.1
```

AdamW 的权重衰减，帮助减少过拟合。

在 `configure_adamw_for_model` 里，二维及以上参数会使用 weight decay；一维参数，比如 norm 的缩放参数，不使用 weight decay。

### `beta1`

默认值：

```python
0.9
```

AdamW 的一阶动量系数。通常不用调。

### `beta2`

默认值：

```python
0.95
```

AdamW 的二阶动量系数。训练语言模型时，`0.95` 是常见选择，比 PyTorch 默认的 `0.999` 更适合一些 Transformer 训练场景。

### `eps`

默认值：

```python
1e-8
```

AdamW 的数值稳定项。通常不用调。

## 设备参数

### `device`

默认值：

```python
"auto"
```

如果是 `"auto"`，脚本会自动选择：

```text
有 CUDA -> cuda
没有 CUDA -> cpu
```

你本机 VS Code 里应该使用 conda 的 `torch` 环境：

```text
C:\Users\84475\miniconda3\envs\torch\python.exe
```

这个环境里的 PyTorch 是 CUDA 版，适合训练。

## 推荐运行方式

快速检查训练循环能不能跑：

```powershell
C:\Users\84475\miniconda3\envs\torch\python.exe .\04_training_loop\train.py --max_steps 10 --eval_interval 5 --eval_iters 2 --checkpoint_interval 0 --batch_size 4 --context_length 64 --d_model 128 --num_layers 2 --num_heads 4 --d_ff 512
```

使用当前默认小模型训练：

```powershell
C:\Users\84475\miniconda3\envs\torch\python.exe .\04_training_loop\train.py
```

如果以后想接近 GPT-3 Small 规模，可以尝试：

```powershell
C:\Users\84475\miniconda3\envs\torch\python.exe .\04_training_loop\train.py --context_length 1024 --d_model 768 --num_layers 12 --num_heads 12 --d_ff 3072 --batch_size 4
```

这会明显增加显存压力。正式训练前，建议先用较小 batch 检查是否能正常跑通。
