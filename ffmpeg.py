import whisper
import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
import subprocess, sys, warnings
warnings.filterwarnings("ignore")

AUDIO_PATH = "托盘贸易-AI初试题.mp3"
# 将变量名稍微改一下，代表全长音频，避免歧义
PROCESSED_PATH = "audio_full.wav" 
OUTPUT_PATH = "转写结果.txt"
NUM_SPEAKERS = 2

# 1. 转换整个音频文件（去掉了 -ss 和 -t 参数）
print("正在使用 FFmpeg 转换全长音频...")
subprocess.run(["ffmpeg", "-i", AUDIO_PATH, 
                "-ar", "16000", "-ac", "1", PROCESSED_PATH, "-y", "-loglevel", "error"])

def fmt_time(s):
    return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d}"

# 2. Whisper 转写
print("加载 Whisper small 模型...")
model = whisper.load_model("small")
# 注意这里改成了 PROCESSED_PATH
result = model.transcribe(PROCESSED_PATH, language="zh", verbose=False) 
segments = result["segments"]
print(f"识别出 {len(segments)} 个语音片段")

# 3. Resemblyzer 说话人嵌入
print("提取说话人特征...")
encoder = VoiceEncoder()
# 注意这里改成了 PROCESSED_PATH
wav_raw, sr = sf.read(PROCESSED_PATH) 
wav_raw = wav_raw.astype(np.float32)

embeddings, valid_idx = [], []
for i, seg in enumerate(segments):
    s, e = int(seg["start"]*sr), int(seg["end"]*sr)
    chunk = wav_raw[s:e]
    if len(chunk) >= sr * 0.5:
        try:
            embeddings.append(encoder.embed_utterance(chunk))
            valid_idx.append(i)
        except: pass

# 4. 聚类分配说话人
print("说话人聚类...")
if len(embeddings) >= NUM_SPEAKERS:
    labels = AgglomerativeClustering(
        n_clusters=NUM_SPEAKERS, metric='cosine', linkage='average'
    ).fit_predict(np.array(embeddings))
else:
    labels = [0] * len(embeddings)

spk_map = dict(zip(valid_idx, labels))
all_labels, last = [], 0
for i in range(len(segments)):
    if i in spk_map: last = spk_map[i]
    all_labels.append(last)

# 5. 合并同一说话人相邻片段
merged, i = [], 0
while i < len(segments):
    seg = segments[i]; spk = all_labels[i]
    text = seg["text"].strip()
    start, end = seg["start"], seg["end"]
    j = i + 1
    while j < len(segments) and all_labels[j] == spk:
        text += segments[j]["text"].strip()
        end = segments[j]["end"]; j += 1
    merged.append({"start": start, "end": end, "speaker": spk+1, "text": text})
    i = j

# 6. 写输出
lines = []
for entry in merged:
    lines.append(f"{fmt_time(entry['start'])} - {fmt_time(entry['end'])} 说话人{entry['speaker']}")
    lines.append(entry["text"])
    lines.append("")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\n✅ 完成！输出已保存至：{OUTPUT_PATH}")