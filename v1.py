"""
改进版说话人分离模块
用法：将此文件中的 diarize_two_speakers() 函数替换掉原脚本中的同名函数。

额外依赖（在原脚本基础上新增）：
    pip install resemblyzer soundfile

改进点：
1. Resemblyzer 神经网络说话人嵌入（替换 MFCC 均值）
2. 滑动窗口分段（替换粗粒度的静音检测分段）
3. AgglomerativeClustering + 余弦距离（替换 KMeans）
4. 时序平滑（消除"说话人乒乓"现象）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str


# ─────────────────────────────────────────────────────────────
# 核心：改进版说话人分离
# ─────────────────────────────────────────────────────────────

def diarize_two_speakers(wav_file: Path, num_speakers: int = 2) -> List[SpeakerSegment]:
    """
    改进版双说话人分离。

    参数：
        wav_file     : 16kHz 单声道 WAV 文件路径
        num_speakers : 说话人数量，默认 2
    返回：
        SpeakerSegment 列表，每段含起止时间与说话人标签
    """
    # ── 1. 加载音频 ──────────────────────────────────────────
    wav_raw, sr = sf.read(str(wav_file), dtype="float32")
    if wav_raw.ndim > 1:
        wav_raw = wav_raw[:, 0]  # 取第一声道
    total_duration = len(wav_raw) / sr

    # ── 2. 滑动窗口切片 ──────────────────────────────────────
    #   窗口大小 2 秒，步长 0.5 秒
    #   每个窗口必须有足够的能量（非静音）才被保留
    WINDOW_SEC = 2.0
    STEP_SEC   = 0.5
    MIN_ENERGY = 1e-4   # 静音阈值（均方根能量）

    window_samples = int(WINDOW_SEC * sr)
    step_samples   = int(STEP_SEC * sr)

    spans: List[Tuple[float, float]] = []
    chunks: List[np.ndarray] = []

    offset = 0
    while offset + window_samples <= len(wav_raw):
        chunk = wav_raw[offset: offset + window_samples]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms >= MIN_ENERGY:                       # 过滤静音帧
            start_sec = offset / sr
            end_sec   = (offset + window_samples) / sr
            spans.append((start_sec, end_sec))
            chunks.append(chunk)
        offset += step_samples

    if not chunks:
        return [SpeakerSegment(0.0, total_duration, "说话人1")]

    # ── 3. Resemblyzer 说话人嵌入 ────────────────────────────
    #   Resemblyzer 期望 16kHz float32 归一化音频
    encoder = VoiceEncoder()
    embeddings: List[np.ndarray] = []
    valid_spans: List[Tuple[float, float]] = []

    for chunk, span in zip(chunks, spans):
        try:
            emb = encoder.embed_utterance(chunk)
            embeddings.append(emb)
            valid_spans.append(span)
        except Exception:
            pass  # 极短片段可能无法提取，跳过

    if not embeddings:
        return [SpeakerSegment(0.0, total_duration, "说话人1")]

    X = np.array(embeddings)   # shape: (N, 256)

    # ── 4. AgglomerativeClustering（余弦距离）─────────────────
    n_clusters = min(num_speakers, len(embeddings))
    if n_clusters == 1:
        raw_labels = np.zeros(len(embeddings), dtype=int)
    else:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        raw_labels = clustering.fit_predict(X)

    # ── 5. 时序平滑（消除"说话人乒乓"）──────────────────────
    #   规则：如果某个窗口的标签与前后邻居都不同，将其归入相邻多数标签。
    #   迭代两次以传播稳定区域。
    labels = raw_labels.copy()
    for _ in range(2):
        for i in range(1, len(labels) - 1):
            if labels[i] != labels[i - 1] and labels[i] != labels[i + 1]:
                labels[i] = labels[i - 1]   # 孤立窗口向左侧靠拢

    # ── 6. 将窗口级标签 → 说话人片段 ─────────────────────────
    #   由于窗口有重叠，同一时间点可能被多个窗口覆盖。
    #   对每个时间点取覆盖它的所有窗口的标签众数（投票）。
    RESOLUTION = 0.1  # 时间轴分辨率（秒）
    n_bins = int(total_duration / RESOLUTION) + 1
    vote_matrix = np.full((n_bins, num_speakers), 0, dtype=int)

    for (start_sec, end_sec), label in zip(valid_spans, labels):
        bin_start = int(start_sec / RESOLUTION)
        bin_end   = int(end_sec   / RESOLUTION)
        if 0 <= label < num_speakers:
            vote_matrix[bin_start:bin_end, label] += 1

    # 对每个时间格投票选出说话人
    bin_labels = np.argmax(vote_matrix, axis=1)

    # ── 7. 合并连续相同说话人的时间格 → 最终片段列表 ─────────
    segments: List[SpeakerSegment] = []
    i = 0
    while i < n_bins:
        label = int(bin_labels[i])
        j = i + 1
        while j < n_bins and bin_labels[j] == label:
            j += 1
        start_sec = i * RESOLUTION
        end_sec   = min(j * RESOLUTION, total_duration)
        speaker   = f"说话人{label + 1}"
        if end_sec > start_sec:
            segments.append(SpeakerSegment(start_sec, end_sec, speaker))
        i = j

    # ── 8. 合并持续时间极短的片段（< 1.5 秒）─────────────────
    segments = _absorb_short_segments(segments, min_dur=1.5)

    return segments


def _absorb_short_segments(
    segments: List[SpeakerSegment],
    min_dur: float = 1.5,
) -> List[SpeakerSegment]:
    """
    把时长 < min_dur 的碎片段并入相邻较长段，
    避免一句话内出现无意义的说话人切换。
    """
    if not segments:
        return []

    changed = True
    while changed:
        changed = False
        merged: List[SpeakerSegment] = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            dur = seg.end - seg.start
            if dur < min_dur and len(segments) > 1:
                # 向左侧相邻段合并
                if merged:
                    prev = merged[-1]
                    merged[-1] = SpeakerSegment(prev.start, seg.end, prev.speaker)
                elif i + 1 < len(segments):
                    nxt = segments[i + 1]
                    merged.append(SpeakerSegment(seg.start, nxt.end, nxt.speaker))
                    i += 2
                    changed = True
                    continue
                changed = True
            else:
                merged.append(seg)
            i += 1
        segments = merged

    # 最后再合并相邻同说话人的片段
    final: List[SpeakerSegment] = []
    for seg in segments:
        if final and final[-1].speaker == seg.speaker:
            final[-1] = SpeakerSegment(final[-1].start, seg.end, seg.speaker)
        else:
            final.append(seg)
    return final


# ─────────────────────────────────────────────────────────────
# 快速测试入口（独立运行验证）
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python diarize_improved.py <wav文件路径>")
        sys.exit(1)
    wav = Path(sys.argv[1])
    segs = diarize_two_speakers(wav)
    print(f"检测到 {len(segs)} 个说话片段：")
    for s in segs:
        dur = s.end - s.start
        print(f"  {s.start:6.2f}s – {s.end:6.2f}s  ({dur:.1f}s)  {s.speaker}")