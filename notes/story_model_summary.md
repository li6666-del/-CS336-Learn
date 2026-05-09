# Story Model Experiment Summary

## Dataset

- Train text: data/TinyStoriesV2-GPT4-train.txt
- Valid text: data/TinyStoriesV2-GPT4-valid.txt
- Train tokens: 541,272,695
- Valid tokens: 5,466,420
- Tokenizer: byte-level BPE, vocab size 10,000
- BPE training sample: first 200,000,000 bytes of train text

## Model

- Architecture: decoder-only TransformerLM
- Layers: 8
- d_model: 512
- Heads: 8
- d_ff: 2048
- Context length: 512
- Parameters: 43,803,136

## Training

- Device: RTX 5090 32GB
- batch_size: 16
- tokens per step: 8,192
- max_steps: 50,000
- total trained tokens: 409,600,000
- effective epochs: 0.76
- optimizer: AdamW
- max learning rate: 3e-4
- min learning rate: 3e-5
- warmup steps: 1,000
- eval interval: 500
- checkpoint interval: 5,000
- total elapsed: 3336.9s (55.6 min)

## Loss

- Final step: 50000
- Final train loss: 1.2363
- Final valid loss: 1.2711
- Best sampled valid loss: 1.2317 at step 47000

The eval loss is estimated from sampled batches, so individual points have noise. The 45k checkpoint was selected for generation because it had strong nearby validation performance and good sample quality.

## Artifacts

- Selected checkpoint: /root/autodl-fs/CS336-Learn/checkpoints_8l512/step_045000.pt
- Final checkpoint: /root/autodl-fs/CS336-Learn/checkpoints_8l512/latest.pt
- Training log: /root/autodl-tmp/CS336-Learn/logs/pretrain_8l512.log
- Samples: /root/autodl-tmp/CS336-Learn/reports/story_model_samples.md
