# Pyannote + 讯飞 LFASR 混合语音转写方案 (v3)

## 1. 简介

本项目是一个高性能、工业级的语音转写与说话人分离工具。与传统的“端到端”云端转写不同，本项目采用了**混合架构**：利用本地 SOTA（State-of-the-Art）声纹模型 **Pyannote 3.1** 进行精确的说话人识别与时间轴切分，再配合**讯飞 LFASR（录音文件转写标准版）**进行高质量的文本还原。

### 核心优势：

- **物理级边界隔离**：通过将不同说话人的音频物理切断，彻底解决了云端 API 在说话人交接处常见的“发语词误判”或“归属混乱”问题。
- **SOTA 级分离精度**：采用 Pyannote Audio 3.1 模型，其声纹聚类精度远超通用 ASR 附带的分离插件。
- **并发处理优化**：支持多线程并发上传音频切片至云端，大幅缩短长音频的处理耗时。
- **预处理自动化**：内置音频清洗逻辑，自动转换采样率、声道并修复可能的解码错误。

## 2. 核心架构逻辑

本脚本的运行分为四个核心阶段：

1. **音频预处理**：利用 `pydub` 将原始音频标准化为 16kHz/单声道/WAV 格式，并可根据需要截取测试片段。
2. **本地说话人分离 (Stage 1)**：调用 `pyannote/speaker-diarization-3.1` 模型，输出精确到毫秒的说话人时间轴（Turn-based segments）。
3. **音频物理切分 (Stage 2)**：根据第一阶段的时间轴，将原始音频切割为多个单一角色的临时 WAV 文件。
4. **云端并发转写 (Stage 3)**：将切割后的切片以 `roleType=0`（单角色模式）并发上传至讯飞 LFASR，识别完成后按时间戳重组结果。

## 3. 环境准备

### 依赖安装

```bash
# 基础依赖
pip install requests pydub

# 深度学习相关 (Pyannote)
pip install pyannote.audio torch torchaudio

# 系统依赖
# 需确保系统已安装 ffmpeg（用于音频处理）
```

### 获取 API 权限

1. **HuggingFace Token**:
   - 前往 [HuggingFace](https://huggingface.co/pyannote/speaker-diarization-3.1) 申请模型访问权限。
   - 在 Settings -> Access Tokens 中创建 Token。
2. **讯飞 LFASR 凭证**:
   - 在[讯飞开放平台](https://www.xfyun.cn/service/lfasr)开通“录音文件转写标准版”服务，获取 `AppID` 和 `SecretKey`。

## 4. 快速开始

### 基础运行

```bash
python v3.py --input "interview.mp3" --output "result.txt"
```

### 常用参数说明

| **参数**          | **说明**                                       | **默认值**        |
| ----------------- | ---------------------------------------------- | ----------------- |
| `--hf-token`      | HuggingFace 访问令牌                           | (需填写)          |
| `--num-speakers`  | 指定说话人数量（如为 2 人播客，填 2 精度最高） | `None` (自动估计) |
| `--workers`       | 讯飞上传的并发线程数                           | `3`               |
| `--gap-threshold` | 同说话人最大间隔合并阈值（秒）                 | `0.8`             |
| `--min-duration`  | 片段最小保留时长，过短将并入邻段（秒）         | `1.5`             |

## 5. 配置与自定义

您可以直接在脚本中修改以下默认值以匹配您的环境：

- **音频截取**：在 `main()` 函数的预处理部分，可以通过 `clean_audio[:60000]` 轻松实现分钟级的快速测试。
- **并发数**：如果您的讯飞账号权限较高，可以适当调大 `--workers` 以获取更快的转写速度。

## 6. 许可证

本项目仅供学术交流与技术研究使用，调用讯飞 API 产生的费用由使用者自行承担。

------

### 引用与致谢

- **Diarization**: [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
- **ASR**: [iFlytek LFASR API](https://www.xfyun.cn/)