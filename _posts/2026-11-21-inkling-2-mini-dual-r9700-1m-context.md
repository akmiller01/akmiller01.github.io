---
layout: post
title:  "What does a million-token context window cost? Running Inkling 2 Mini on a dual R9700"
author: alex_miller
date:   2026-11-21
categories: blog
---

In [my last post](/blog/2026/09/07/dual-r9700-llama-cpp-and-vllm.html){:target="_blank"}, I took my dual Radeon AI PRO R9700 rig from a painful 19 tokens per second up to 185 on Qwen 3.8 27B. The machine has been running that Qwen setup ever since, but this week I pointed it at a new model: **Inkling 2 Mini**, whose support landed on llama.cpp's master branch just yesterday. This is a follow-up to that story: what it takes to get a one-million-token context window out of two 32 GB cards, what that window costs in speed, and what a few hours of flag tuning buys back.

If you want to skip the build and the 1M context setup and jump straight to the performance results, click [here](#results).

## The build

Model support landed on master in the commit [`29b52d398`](https://github.com/ggml-org/llama.cpp/commit/29b52d398b294509746ec0a3f28948a3107c0213){:target="_blank"} ("model : Inkling 2 Mini (#32694)"), so I built llama.cpp from that commit with the same `gfx1201`-targeted ROCm/HIP recipe as last time:

```
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 29b52d398b294509746ec0a3f28948a3107c0213
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S . -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1201 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

I got the GGUFs from ggml-org's HuggingFace hub: [Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF){:target="_blank"}. I used the Q8_0 variant (27.8 GiB) for the two-card runs and the IQ4_NL variant (17.6 GiB) for the single-card runs, plus the Q8_0 `mmproj` (346 MiB) for vision.

Inkling 2 Mini has an interesting architecture. It has 24 physical layers that expand into 48 cache slots, because a core of layers is looped over three times, plus 20 sliding-window layers (window 1024). The KV cache is correspondingly small (about 17.4 KiB per token for the full-attention layers), but that loop structure has consequences for how llama.cpp can split the model across GPUs, as I'll show next.

## A 1M context window on two R9700s

Can a 27.8 GiB Q8_0 model plus a one-million-token context actually fit in 64 GiB of VRAM? Yes, and with room to spare. The KV cache is the good news here: at about 17.4 KiB per token, a 1M context costs about 17.4 GiB. Add the 27.8 GiB of weights and the 346 MiB vision projector, and the whole thing fits in 64 GiB, but only if the weights and KV are spread across both cards. And that's where the first snag is.

llama.cpp's default multi-GPU mode is a layer split, but the default free-memory split ratio of [0.5, 1.0] put every physical layer of Inkling on GPU 0; the weights overflowed before the KV cache was even allocated. (I suspect the loop-expanded slots confuse the heuristic.) The fix is an explicit split point: `-ts 1,2.2` sets the split to [0.3125, 1.0], which lands slots 0-15 on GPU 0 and 16-48 on GPU 1.

And a layer split is the *only* option. Tensor split, the mode that made Qwen fast, is explicitly unsupported for this architecture (the model's `inkling2_loop` is on the unsupported list in `src/llama-arch.cpp`), and row split in this build is only implemented for SYCL and Hexagon, not ROCm.

The full command:

```
llama-server \
  -m ~/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q8_0.gguf \
  --mmproj ~/models/Inkling-2-Mini-GGUF/mmproj-Inkling-2-Mini-Q8_0.gguf \
  -ngl 999 \
  -sm layer \
  -ts 1,2.2 \
  -fa on \
  -b 2048 \
  -ub 16 \
  -lm none \
  -ctk q8_0 -ctv q8_0 \
  -c 1048576 \
  -a Inkling-2-Mini-Q8_0 \
  --port 8080
```

Three notes. `-lm none` is still required on this machine (memory-mapping hangs at load, same as last time) and needs about 28 GiB of free host RAM while the weights load. `-fa on` is not optional: a quantized V cache *requires* flash-attention in this build. And `-ub 16` is the victim of the flash-attention scratch buffer, which is about 199 bytes per ubatch x ctx pair; at a 1M context, any larger ubatch would blow the card.

It boots: about 31 GiB used on GPU 0, about 29 GiB on GPU 1.

## The performance hit

What does that 1M window cost? Prompt processing. (The numbers below come from my agent-workload benchmark: a realistic multi-turn tool-calling conversation with an ~8.7k-token "agent" prompt and an ~35k-token "long" prompt, 3 reps each, mean +/- stddev, temperature 0.7, max 384 completion tokens. Each rep carries a unique nonce so the server can't serve a cached prompt.)

<div class="overflow-table" markdown="1">

| Profile | Prompt (tok/s) | Generation (tok/s) |
| :---- | :---: | :---: |
| agent (~8.7k) | 158.5 | 16.41 |
| long (~35k) | 142.98 | 15.03 |

</div>

Generation at about 16 tokens per second is in the same league as the 19.2 tokens per second I started with in the Qwen post: the 27.8 GiB of Q8_0 weights are simply a lot to move per token, and a layer split means the two cards take turns rather than working in parallel. Prompt processing is worse: 158.5 tokens per second, versus Qwen's 1,458.

## Tuning

The key insight is that prefill speed depends on the *actual* prompt length, not the allocated context. The only real cost of a large context is VRAM: the KV cache and the flash-attention scratch buffer (about 199 bytes x ubatch x context) both scale with it, and together they force a smaller ubatch, and the ubatch is what governs how much of the prompt I process in parallel.

So I ran the single-card IQ4_NL variant (17.6 GiB, fits on one R9700) through a series of context/ubatch combinations, benchmarking only the agent profile to keep iteration time short:

<div class="overflow-table" markdown="1">

| Config (IQ4_NL, 1-GPU) | Prompt (tok/s) | Generation (tok/s) |
| :---- | :---: | :---: |
| 524288 / ub 32 (default) | 319.03 | 32.75 |
| 65536 / ub 128 | 502.97 | 32.74 |
| 65536 / ub 256 | 471.32 | 32.50 |
| 49152 / ub 256 | 473.26 | 32.70 |
| 49152 / ub 128 | 497.21 | 32.63 |
| 49152 / ub 128, q4 KV | 498.09 | 33.00 |

</div>

Two findings. First, ubatch 128 is the sweet spot: going 32 to 128 buys +58% prompt processing, but 128 to 256 regresses by 6% as the flash-attention scratch grows. Second, generation is weight-bandwidth-bound (about 32.7 tokens per second on IQ4_NL) and doesn't move with any of these flags; quantizing the KV to q4 is neutral at an 8.7k context (it would save bandwidth at longer contexts, but the VRAM was never the constraint here).

64K of context is more than enough for the benchmark workloads, but I wanted headroom for real work, so I checked the 256K context. It still fits ubatch 128 on a single card (about 29 GiB used), and it matches the 64K speed to within the noise (496.37 vs 498.07 on the agent profile, 337.93 vs 338.27 on the long profile).

The same extrapolation to the dual-GPU Q8_0: at a 256K context, each card pays about 6.2 GiB of flash-attention scratch, and from the verified 1M anchor that puts GPU 0 at about 29 GiB and GPU 1 at about 24 GiB. It boots, and the numbers:

<div class="overflow-table" markdown="1">

| Config (Q8_0, 2-GPU layer split) | Prompt (tok/s) | Generation (tok/s) |
| :---- | :---: | :---: |
| 1048576 / ub 16 | 158.5 / 142.98 | 16.41 / 15.03 |
| 262144 / ub 128 | 417.77 / 312.59 | 18.40 / 16.55 |

</div>

So trading a 1M window for a 256K one buys +163% (2.6x) prompt processing and +12% generation.

## Results

For the full picture, here is where the performance landed for all the models at a 256K context window with the same benchmark:

<div class="overflow-table" markdown="1">

| Model / stack | Agent prompt (tok/s) | Agent generation (tok/s) | Long prompt (tok/s) | Long generation (tok/s) |
| :---- | :---: | :---: | :---: | :---: |
| Qwen 3.8 27B Q8_0, llama.cpp (tensor split + MTP) | 1,581.6 | 61.12 | 1,506.37 | 58.18 |
| Qwen 3.8 27B MXFP4, vLLM + Radiance (dflash) | 5,280.22 | 194.2 | 4,862.2 | 177.45 |
| Inkling 2 Mini Q8_0, llama.cpp (2-GPU layer split, ub 128) | 417.77 | 18.40 | 312.59 | 16.55 |
| Inkling 2 Mini IQ4_NL, llama.cpp (1-GPU, ub 128) | 496.37 | 32.68 | 337.93 | 30.76 |

</div>

Two observations. First, the single-card IQ4_NL beats the dual-GPU Q8_0 on both prompt processing and generation: the 17.6 GiB weight footprint is just less to move per token, and one card doesn't have to wait on the other. And second, Inkling 2 Mini sits 3-5x behind Qwen 3.8 27B on both llama.cpp and vLLM: there's no speculative decoding for this architecture yet, and the layer split (the only mode this architecture supports) doesn't pool the two cards' bandwidth the way a tensor split does.

## Conclusions

For day-to-day work, I'm sticking with Qwen 3.8 27B on the Radiance vLLM stack; 194 tokens per second of generation is hard to argue with, and 256K of context is plenty. But Inkling 2 Mini is a model to keep an eye on for its one-million-token context window and looped architecture.

To be honest, 16-18 tokens per second of generation on a dual R9700 is not where I'd like to be, but the architecture is new enough that I'd expect the tuning story to still be moving. I'm curious what other people are seeing: is this the norm for Inkling 2 Mini at Q8_0, or am I missing something? If you've run it, I'd love to hear your numbers.
