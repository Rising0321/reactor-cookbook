# LiveAvatar 实机测试记录 — 2026-09-16

## 结论

**音频更正：** 下述历史 GPU 测试的 `audio.wav` 存在 16/48 kHz 采样率不匹配，
不应作为音频质量通过的证据；`take.mp4` 使用原始上传音频，掩盖了此问题。
v0.2.0 已修复输出重采样，另做 CPU Runtime → SDK 回归，见 AUDIO_FIX_REPORT.md。
本次修复后没有重跑 GPU 推理；历史产物保留不覆盖。

第一阶段的真实模型加载、上传输入、连续自回归推理和 Reactor SDK 音视频接收已跑通。
先完成 3 个 clip（141 帧），再用最终代码完成 6 个 clip（285 帧）。
最终成片为 **384×704、25 FPS、11.4 秒**。没有设置默认图像；测试客户端显式上传
官方 `fashion_blogger.jpg` 和 `fashion_blogger.wav`，并传入讲话/自然手势 Prompt。

## 真实产物

目录：`/opt/dlami/nvme/.cache_hf/reactor_registry/liveavatar-stage1/real-test-20260916-final/`

- `take.mp4`：SDK 收到的真实生成帧，按原生 25 FPS 与上传的音频合成的模型时间线预览。
- `video.mp4`：SDK 收到的 285 帧，不含音轨。
- `audio.wav`：SDK 实际接收的 PCM，保留推理等待期间 Runtime 插入的静音。
- `messages.json`：上传接受、开始、逐 clip 完成及正常结束的消息记录。
- `contact-sheet.jpg`：覆盖 11.4 秒的时序抽帧。
- `late-boundaries.jpg`：后两处 clip 交界附近的连续帧。

服务日志：`/opt/dlami/nvme/.cache_hf/reactor_registry/liveavatar-stage1/final-service.log`

客户端日志：`/opt/dlami/nvme/.cache_hf/reactor_registry/liveavatar-stage1/final-client.log`

## 检查结果

| 检查 | 结果 |
| --- | --- |
| 单卡 | 只使用开始时空闲的 GPU 0，其他卡未使用 |
| 真实权重 | Wan2.2-S2V-14B + 官方四步 LiveAvatar 蒸馏 LoRA，成功加载/合并 |
| 显存 | 观察到约 94,724 MiB（92.5 GiB）；非高频采样的严格峰值 |
| KV Cache | 原生四份采样步缓存、默认 clip 长度和 73 帧 motion history 未缩短 |
| 输入 | 显式上传图像、音频和 Prompt，未启用默认图像 |
| 分块 | 单次上游 generate 保持历史，每轮交付一个 clip；45 + 5×48 = 285 帧 |
| SDK | 完整收到 285 帧与音轨，`generation_ended.reason=complete` |
| 时序画质 | 抽帧及交界连续帧检查未见明显身份漂移、背景突变或跨段跳切 |
| 动作 | 张嘴讲话、眨眼、轻微头部动作和手势整体合理；快速手势存在软化/运动模糊 |
| 回归 | 12 项 CPU 测试、schema 导出和 Ruff 检查通过 |

这里的视觉结论来自时序抽帧检查，不等同于正式口型同步评分。没有声称达到完美
逐音素对齐，也未对 pose 条件、多人物或任意输入做全面验收。

## 本次修复

1. 补齐缺失的 **FlashAttention 2.8.3** 原生依赖。首次真实推理在上游交叉注意力的
   `FLASH_ATTN_2_AVAILABLE` 断言失败；使用官方匹配 wheel 后通过。没有更换为 FA4。
2. 上游异常保留 traceback，空字符串的异常用类型名报告，不再只有空的失败原因。
3. 外部 `stop` 命令的 Python 方法改名为 `stop_take`，避免覆盖 Runtime 的同步停机方法。
4. 修正测试视频导出：生成帧按模型时间线压紧时，不应搭配含墙钟等待静音的接收音轨。
   因此 `take.mp4` 使用上传音频，真实接收 PCM 单独保留，明确区分两者。

## 已知限制

当前是调试用第一阶段，**不是实时流畅播放版本**。推理慢于 25 FPS 播放速度，等待
下一段时 Runtime 会补静音，客户端可能表现为画面等待。成片流畅不代表实时链路无等待。
FA4、CUDA graph、编译、进一步细化 VAE 输出等加速均未实施。

测试结束后关闭了本次服务以释放显卡；未停止其他人的任务。
重新启动和上传测试命令见 `README.md`。
