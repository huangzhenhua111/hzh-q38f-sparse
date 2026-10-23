# Qwen3.8-Flash-Next 缩小版"全套" sparse：训练与评测代码（q38f）

黄振华，2026-10-23。按 https://huggingface.co/Qwen/Qwen3.8-Flash-Next 文本部分等比例缩小，在 C1=3e17 上扫 5 个尺寸找底（16k 词表 + seq 2048）。
结论见 `RESULT.md`（底 h384×10，bpb 1.0802；比同设置 Qwen3.5 sparse 底 1.1071 低 0.027），实现假设见 `ASSUMPTIONS.md`。

**注意：激活比例不满足赛规。** 赛题 PDF 要求非词表口径 Na/N ≥ 10%；这里 256 专家选 5，5 个点 Na/N 只有 7.2%–7.6%（N 不含 N-gram 表；若 N-gram 表算进 N，底点约 5.4%）。原版 Qwen3.8-Flash-Next 本身也只有 6B/125B≈4.8%。要合规需减少专家数（如 128 选 5 ≈14%，若 N-gram 表也计入则需 64 选 5 或缩小 N-gram 表），改后需重新扫 U。

## 结构（相对 Qwen3.5 sparse 的改动）
- 层布局：每 4 层 3 层 Gated DeltaNet + 1 层 **QSA**（块稀疏注意力：块 64、每个 query 选 8 块=512 token，lightning indexer 4 头 MQA，对主注意力做 KL 蒸馏，前 1% 步 dense warmup；用 flex_attention）
- **MoE**：256 专家选 5 + 1 共享，专家宽 = 共享宽 = h/4（模型卡是 512 选 10，在 4090 上吞吐不足一半，退到 256/5）
- **N-gram Embedding**：bigram/trigram 各一张哈希表（行数 ≈0.2×非词表总参数/h），查表求和→RMSNorm→门控后加到第 2 层输入；不跨文档
- **4 路分支残差**（mHC：H_pre/H_post/H_res，Sinkhorn 投双随机，低秩瓶颈 r=h/8，初始化等价普通残差）
- **Muon**（隐层矩阵与专家逐片；Nesterov 0.95、NS 5 步、按 Moonlight 对齐 update RMS，沿用 AdamW 的 lr 规则）；嵌入、N-gram 表、router、norm 等用 AdamW
- **1 层 MTP**（预测 x_{t+2}，loss 系数 0.3，FLOPs 计入预算，评测只用主头）
- 16k 自训词表 bpe16k_v2（绑定嵌入）、seq 2048、batch 524288 token、cosine→0.1

## 目录
| 路径 | 内容 |
|---|---|
| `code/train_q38f.py` | 训练（单卡，ckpt 续跑 `--resume 1`，结束导出 HF 目录 + extras.pt 到 `runs/<run>/hf`） |
| `code/q38f_model.py` | QSA、分支残差、MTP 等模型改动（挂在 transformers 的 qwen3_5_moe 上） |
| `code/ngram_q38f.py`、`code/muon_q38f.py` | N-gram Embedding、Muon |
| `code/plan_q38f.py`、`code/gen_rows.py` | 尺寸→配置（头数、专家宽、N-gram 行数、训练 token） |
| `code/count_flops_q38f.py` | 参数量（激活/总、N-gram、MTP 分列）与训练 FLOPs（QSA 按实际选中 token，indexer/蒸馏/mHC/MTP 全计入） |
| `code/eval_q38f.py`、`code/eval_local_q38f.sh` | 官方 eval.py 改的评测（加载自定义模块；bpb 口径不变），单卡 |
| `code/test_q38f.py`、`code/check_export.py` | 单元测试、导出一致性检查；`bench_q38f.py`/`prof_q38f.py` 测速 |
| `launch/run_one.sh` | 一个点：训练→评测→FLOPs 核算→RUN_ALL_DONE；`worker.sh` 按队列跑；`eval_only.sh` 只补评测 |
| `run_configs/q38f_<h>x<L>/` | 5 个点的 run_config / summary / flops / eval 结果 json |
| `tokenizer/bpe16k_v2/` | 16k 词表 |
| `env/` | 依赖与建环境（同 hzh_q35_sparse：torch 2.7.1+cu126、transformers 5.19、fla 0.5.2） |
| `fig/q38f_c1.png` | U 图（同图有 Qwen3.5 sparse 16k 6 点与 dense B 线） |

## 怎么起
脚本里的路径写的是原始位置（`S=/home/hzh/scicomp26`：`.venv_q35/`、`data/corpus_bpe16k_big/train.bin`、`runs/`、`eval/`），在别处用请改 `launch/run_one.sh` 开头的 `S`/`Q` 和 `code/eval_local_q38f.sh` 里的 python 路径。
```bash
# 一个点（UUID 指定卡；参数: h L lr micro_bs ngram_rows max_steps E K）
bash launch/run_one.sh GPU-xxxx 384 10 2.5e-3 8 160768 0 256 5
# 单测
cd code && python test_q38f.py
```
5 个点的参数：256×8 lr3.5e-3 mb16 ngram 84992；320×8 3e-3 mb16 106496；384×10 2.5e-3 mb8 160768；448×12 2.5e-3 mb8 224256；512×12 2.5e-3 mb8 254976（均 256 选 5）。
原始位置：4090a `/home/hzh/scicomp26/q38f/`（代码、启动脚本）、`/home/hzh/scicomp26/runs/q38f_*`（权重与日志）。
