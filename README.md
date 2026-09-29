# hear-your-voice 🎙️

**多说话人录音转写与说话人分离（Speaker Diarization）实验项目**

把一段多人对话录音（会议、播客、访谈）转成带时间戳和说话人标签的文本。项目围绕**科大讯飞语音转写 API** 与**本地开源模型**两条技术路线，逐步演进出了四个版本，用于对比不同方案在中文多人对话场景下的效果与成本。

---

## 版本对比

| 版本 | 方案 | 说话人分离 | 转写 | 特点 |
|:---|:---|:---|:---|:---|
| **`v1.py`** | 本地全离线 | Resemblyzer 声纹嵌入 + 层次聚类 | 依赖宿主脚本 | 以模块形式提供 `diarize_two_speakers()`，可整段替换旧实现 |
| **`ffmpeg.py`** | 本地全离线 | Resemblyzer + 聚类 | OpenAI Whisper (`small`) | 单文件端到端脚本，无需任何 API 费用 |
| **`v2.py`** | 讯飞 LFASR **标准版** | 讯飞云端角色分离 | 讯飞云端 | 完整闭环 + 边界发语词智能修正 |
| **`v2_llm.py`** | 讯飞转写 **大模型版** | 讯飞云端角色分离 | 讯飞大模型 | 换用大模型接口，鉴权机制不同（HMAC-SHA1 动态签名） |
| **`v3.py`** | **混合架构** ⭐ | 本地 Pyannote 3.1 | 讯飞 LFASR 标准版 | 精度最高，见下 |

### v3 的核心思路

`v3.py` 不把整段音频丢给云端，而是**先本地分离、再云端转写**：

1. **音频预处理** — `pydub` 统一转成 16kHz / 单声道 / WAV，规避解码报错
2. **本地说话人分离** — `pyannote/speaker-diarization-3.1` 输出毫秒级说话人时间轴
3. **音频物理切分** — 按时间轴把音频切成单一说话人的临时切片
4. **云端并发转写** — 切片以 `roleType=0`（单角色模式）并发上传讯飞，按时间戳重组

因为每个切片里**只可能有一个说话人**，从物理上杜绝了云端 API 在说话人交接处把下一位的发语词误判给上一位的老问题——这是混合架构相对纯云端的最大优势。

---

## 环境准备

```bash
pip install -r requirements.txt
```

还需系统安装 **ffmpeg** 并确保在 `PATH` 中（`v3.py` 与 `ffmpeg.py` 依赖）。

> `pyannote.audio` + `torch` 体积较大；若只用 `v2.py` / `v2_llm.py`，仅需 `pip install requests`。

---

## 凭证配置

⚠️ **本项目所有凭证均通过环境变量读取，仓库内不含任何真实密钥。** 请勿把自己的密钥写回代码后提交。

需要两类凭证：

### 1. 讯飞开放平台

在[讯飞开放平台](https://www.xfyun.cn/)开通对应服务后获取。注意**标准版**与**大模型版**是两套不同的凭证：

| 环境变量 | 用途 | 用于 |
|:---|:---|:---|
| `XFYUN_APPID` | 应用 ID | 全部云端版本 |
| `XFYUN_SECRET_KEY` | 标准版 secretkey | `v2.py`、`v3.py` |
| `XFYUN_API_KEY` | 大模型版 accessKeyId | `v2_llm.py` |
| `XFYUN_API_SECRET` | 大模型版 apiSecret | `v2_llm.py` |

### 2. HuggingFace Token（仅 `v3.py`）

需先在 [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) 页面申请模型访问权限，再到 Settings → Access Tokens 创建。

| 环境变量 | 用途 |
|:---|:---|
| `HF_TOKEN` | 拉取 Pyannote 模型权重 |

### 设置方式

**Windows PowerShell**（当前会话有效）：

```powershell
$env:XFYUN_APPID      = "你的AppID"
$env:XFYUN_SECRET_KEY = "你的SecretKey"
$env:HF_TOKEN         = "hf_xxx"
```

**Linux / macOS / Git Bash**：

```bash
export XFYUN_APPID="你的AppID"
export XFYUN_SECRET_KEY="你的SecretKey"
export HF_TOKEN="hf_xxx"
```

需要持久化可写入系统环境变量，或使用 `.env` 配合 `python-dotenv`（`.env` 已在 `.gitignore` 中）。

> 所有参数也都可以通过命令行传入，如 `--appid xxx --secret-key xxx`；未提供时会给出明确提示而非静默失败。

---

## 快速开始

### v3 —— 混合架构（推荐）

```bash
python v3.py --input "会议录音.mp3" --output "result.txt" --num-speakers 2
```

| 参数 | 说明 | 默认值 |
|:---|:---|:---|
| `--input` | 输入音频（mp3/wav/m4a…） | `托盘贸易-AI初试题.mp3` |
| `--output` | 输出 txt 路径 | `result.txt` |
| `--num-speakers` | 说话人数量，双人播客填 `2` 精度最高 | 自动估计 |
| `--workers` | 讯飞上传并发线程数 | `3` |
| `--gap-threshold` | 同说话人片段合并间隔阈值（秒） | `0.8` |
| `--min-duration` | 片段最小保留时长（秒） | `1.5` |

### v2 / v2_llm —— 纯云端

```bash
python v2.py     --input "录音.mp3" --output result.txt --role-num 2
python v2_llm.py --input "录音.mp3" --output result.txt --role-num 2
```

`--role-num` 填 `0` 为盲分，明确人数（如 `2`）能显著提升分离精度。

两个版本都支持边界发语词智能修正（默认开启），如产生误伤可用 `--no-fix-fillers` 关闭；加 `--verbose` 可查看每条修正记录，`--dump-raw out.json` 可保留讯飞原始返回便于排查。

### ffmpeg.py —— 全离线

```bash
python ffmpeg.py
```

改脚本顶部的 `AUDIO_PATH` / `NUM_SPEAKERS` 即可。首次运行会下载 Whisper `small` 模型。

### v1.py —— 作为模块使用

`v1.py` 不是独立脚本，把其中的 `diarize_two_speakers()` 替换进你的宿主脚本即可：

```bash
pip install resemblyzer soundfile scikit-learn
```

---

## 输出格式

```text
00:00:00 - 00:00:16 说话人1
Hello大家好，欢迎收听今天的播客，然后咱们今天还是聊托盘贸易啊……

00:00:16 - 00:00:20 说话人2
听起来很有意思啊，那我们就直接开始吧。
```

完整示例见 [`result.txt`](result.txt)。

---

## 文件说明

```text
├── v1.py                        # 本地声纹分离模块（Resemblyzer）
├── ffmpeg.py                    # 本地 Whisper + Resemblyzer 端到端脚本
├── v2.py                        # 讯飞 LFASR 标准版
├── v2_llm.py                    # 讯飞转写大模型版
├── v3.py                        # Pyannote + 讯飞 混合架构
├── v2_llm_readme.md             # 大模型版详细说明
├── v3_readme.md                 # 混合架构详细说明
├── 录音文件转写（标准版）.md      # 讯飞标准版接口笔记
├── 日志文档.md                   # 开发过程记录
└── result.txt                   # 转写输出示例
```

---

## 注意事项

- **费用**：调用讯飞 API 按用量计费，由使用者自行承担。`v3.py` 会把音频切片后逐片上传，请注意用量。
- **文件上限**：讯飞单文件 500MB、时长 5 小时，脚本内已做前置校验。
- **密钥安全**：`.gitignore` 已屏蔽 `.env` 与 `credentials.json`。提交前请确认 `git status` 中没有凭证文件。
- **模型缓存**：Pyannote 与 Whisper 权重动辄数 GB，已加入 `.gitignore`。

## 引用与致谢

- **说话人分离**：[pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
- **声纹嵌入**：[resemble-ai/Resemblyzer](https://github.com/resemble-ai/Resemblyzer)
- **语音识别**：[OpenAI Whisper](https://github.com/openai/whisper) · [讯飞开放平台](https://www.xfyun.cn/)

## 许可证

本项目仅供学术交流与技术研究使用。
