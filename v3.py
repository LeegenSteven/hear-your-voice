from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

def _import_pyannote():
    try:
        from pyannote.audio import Pipeline
        return Pipeline
    except ImportError:
        sys.exit(
            "❌ 缺少 pyannote.audio\n"
            "   pip install pyannote.audio torch torchaudio"
        )

def _import_pydub():
    try:
        from pydub import AudioSegment
        return AudioSegment
    except ImportError:
        sys.exit(
            "❌ 缺少 pydub\n"
            "   pip install pydub\n"
            "   同时确保 ffmpeg 已安装"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 1 — Pyannote Speaker Diarization
# ══════════════════════════════════════════════════════════════════════════════

def run_diarization(
    audio_path: Path,
    hf_token: str,
    num_speakers: Optional[int],
) -> List[Tuple[float, float, str]]:
    """
    返回 [(start_sec, end_sec, speaker_label), ...]，已按时间排序。

    num_speakers=None  → Pyannote 自动估计人数（最慢，精度略低）
    num_speakers=2     → 固定双人，推理更快且边界更准
    """
    Pipeline = _import_pyannote()

    print("  [Pyannote] 加载模型 pyannote/speaker-diarization-3.1 ...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    # 优先使用 GPU
    try:
        import torch
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
            print("  [Pyannote] 使用 CUDA 加速")
        elif torch.backends.mps.is_available():
            pipeline = pipeline.to(torch.device("mps"))
            print("  [Pyannote] 使用 Apple MPS 加速")
        else:
            print("  [Pyannote] 使用 CPU（较慢，请耐心等待）")
    except Exception:
        pass

    print(f"  [Pyannote] 开始推理：{audio_path.name} ...")
    t0 = time.time()

    kwargs: dict = {"audio": str(audio_path)}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(kwargs["audio"], **{
        k: v for k, v in kwargs.items() if k != "audio"
    })

    elapsed = time.time() - t0
    print(f"  [Pyannote] 推理完成，耗时 {elapsed:.1f}s")

    segments: List[Tuple[float, float, str]] = []
    
    # 兼容新老版本的 pyannote.audio 返回值
    if hasattr(diarization, "speaker_diarization"):
        # 新版本：返回的是 DiarizeOutput 对象
        annotation = diarization.speaker_diarization
    else:
        # 老版本：直接返回 Annotation 对象
        annotation = diarization
        
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

    segments.sort(key=lambda x: x[0])
    return segments


# ══════════════════════════════════════════════════════════════════════════════
#  后处理：合并 & 重标说话人编号
# ══════════════════════════════════════════════════════════════════════════════

def merge_diarization(
    segments: List[Tuple[float, float, str]],
    gap_threshold_s: float = 0.8,
    min_duration_s: float  = 1.5,
) -> List[Tuple[float, float, str]]:
    """
    两步合并：
      1. 同一说话人、间隔 < gap_threshold_s → 合并
      2. 合并后仍 < min_duration_s 的短片段 → 并入时间最近的邻居
         （极短片段送 LFASR 往往返回空结果）
    """
    if not segments:
        return []

    # Step 1: gap 合并
    merged: List[Tuple[float, float, str]] = []
    cur_s, cur_e, cur_spk = segments[0]
    for s, e, spk in segments[1:]:
        if spk == cur_spk and (s - cur_e) <= gap_threshold_s:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e, cur_spk))
            cur_s, cur_e, cur_spk = s, e, spk
    merged.append((cur_s, cur_e, cur_spk))

    # Step 2: 短片段吸收（向后合并）
    result: List[Tuple[float, float, str]] = []
    for s, e, spk in merged:
        if (e - s) < min_duration_s and result:
            ps, pe, pspk = result[-1]
            result[-1] = (ps, max(pe, e), pspk)
        else:
            result.append((s, e, spk))

    return result


def relabel_speakers(
    segments: List[Tuple[float, float, str]],
) -> Tuple[List[Tuple[float, float, int]], Dict[str, int]]:
    """
    把 Pyannote 的 SPEAKER_00/SPEAKER_01 → 说话人1/2（按首次出现顺序）
    返回 (重标后的segments, label_map)
    """
    label_map: Dict[str, int] = {}
    relabeled = []
    for s, e, spk in segments:
        if spk not in label_map:
            label_map[spk] = len(label_map) + 1
        relabeled.append((s, e, label_map[spk]))
    return relabeled, label_map


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 2 — 音频切割
# ══════════════════════════════════════════════════════════════════════════════

def slice_audio(
    audio_path: Path,
    segments: List[Tuple[float, float, int]],
    tmp_dir: Path,
) -> List[Tuple[float, float, int, Path]]:
    """
    切割音频为每段一个 wav 文件，返回 (start, end, speaker, wav_path)。
    统一转为 16kHz 单声道 wav（讯飞 LFASR 最友好的格式）。
    """
    AudioSegment = _import_pydub()

    print(f"  [切割] 加载音频 {audio_path.name} ...")
    audio = AudioSegment.from_file(str(audio_path))
    audio = audio.set_frame_rate(16000).set_channels(1)

    result = []
    for i, (s, e, spk) in enumerate(segments):
        start_ms = int(s * 1000)
        end_ms   = int(e * 1000)
        chunk    = audio[start_ms:end_ms]
        wav_path = tmp_dir / f"seg_{i:04d}_spk{spk}.wav"
        chunk.export(str(wav_path), format="wav")
        result.append((s, e, spk, wav_path))
        print(f"  [切割] {i+1}/{len(segments)}  说话人{spk}  "
              f"{_fmt(s)} → {_fmt(e)}  ({(e-s):.1f}s)", end="\r")

    print(f"\n  [切割] 完成，共 {len(result)} 段")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 3 — 讯飞 LFASR（逐段转写，并发）
# ══════════════════════════════════════════════════════════════════════════════

LFASR_HOST = "https://raasr.xfyun.cn"
UPLOAD_API = "/v2/api/upload"
GET_API    = "/v2/api/getResult"


def _make_signa(appid: str, secret_key: str, ts: int) -> str:
    base_string = (appid + str(ts)).encode("utf-8")
    md5_hex     = hashlib.md5(base_string).hexdigest()
    digest      = hmac.new(
        secret_key.encode("utf-8"),
        md5_hex.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def _upload(appid: str, secret_key: str, wav_path: Path) -> str:
    ts       = int(time.time())
    signa    = _make_signa(appid, secret_key, ts)
    file_size = wav_path.stat().st_size
    params   = {
        "appId": appid, "signa": signa, "ts": ts,
        "fileName": wav_path.name, "fileSize": file_size,
        "duration": 200,
        "language": "cn",
        "roleType": 0,   # ← 关键：不开角色分离，说话人由 Pyannote 已定
    }
    with wav_path.open("rb") as f:
        resp = requests.post(
            LFASR_HOST + UPLOAD_API,
            params=params,
            headers={"Content-Type": "application/octet-stream"},
            data=f, timeout=300,
        )
    resp.raise_for_status()
    body = resp.json()
    if str(body.get("code")) != "000000":
        raise RuntimeError(f"上传失败: {body}")
    return body["content"]["orderId"]


def _poll(appid: str, secret_key: str, order_id: str,
          poll_interval: int, max_wait: int) -> dict:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        ts    = int(time.time())
        signa = _make_signa(appid, secret_key, ts)
        resp  = requests.post(
            LFASR_HOST + GET_API,
            params={"appId": appid, "signa": signa, "ts": ts,
                    "orderId": order_id, "resultType": "transfer"},
            timeout=30,
        )
        resp.raise_for_status()
        body   = resp.json()
        status = body["content"]["orderInfo"].get("status")
        if status == 4:
            return body
        if status == -1:
            raise RuntimeError(f"LFASR 转写失败: {body}")
        time.sleep(poll_interval)
    raise RuntimeError(f"LFASR 超时 {max_wait}s")


def _extract_text(body: dict) -> str:
    """从 LFASR 返回体中提取纯文本。"""
    order_result = json.loads(body["content"]["orderResult"])
    tokens: List[str] = []
    for item in order_result.get("lattice2", []):
        st = item.get("json_1best", {}).get("st", {})
        for rt in st.get("rt", []):
            for ws in rt.get("ws", []):
                for cw in ws.get("cw", []):
                    w = cw.get("w", "")
                    if w:
                        tokens.append(w)
    return "".join(tokens).strip()


def transcribe_one(
    appid: str,
    secret_key: str,
    entry: Tuple[float, float, int, Path],
    poll_interval: int,
    max_wait: int,
    idx: int,
    total: int,
) -> Tuple[float, float, int, str]:
    """单段转写（供线程池调用）。"""
    s, e, spk, wav_path = entry
    tag = f"[LFASR {idx+1}/{total}]"
    try:
        order_id = _upload(appid, secret_key, wav_path)
        print(f"  {tag} 说话人{spk} {_fmt(s)}-{_fmt(e)} 已上传 orderId={order_id}")
        body = _poll(appid, secret_key, order_id, poll_interval, max_wait)
        text = _extract_text(body)
        display = (text[:30] + "…") if len(text) > 30 else text
        print(f"  {tag} ✅ 完成 → {display}")
        return (s, e, spk, text)
    except Exception as exc:
        print(f"  {tag} ⚠️ 失败：{exc}，跳过此段")
        return (s, e, spk, "")


def transcribe_all(
    appid: str,
    secret_key: str,
    slices: List[Tuple[float, float, int, Path]],
    workers: int,
    poll_interval: int,
    max_wait: int,
) -> List[Tuple[float, float, int, str]]:
    """并发转写所有片段，返回按时间排序的结果。"""
    results: List[Tuple[float, float, int, str]] = []
    total = len(slices)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                transcribe_one,
                appid, secret_key, entry,
                poll_interval, max_wait, i, total
            ): i
            for i, entry in enumerate(slices)
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda x: x[0])
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  输出
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(sec: float) -> str:
    total = max(0, int(sec))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_output(
    results: List[Tuple[float, float, int, str]],
    output_file: Path,
) -> None:
    with output_file.open("w", encoding="utf-8") as f:
        for s, e, spk, text in results:
            if not text:
                continue
            f.write(f"{_fmt(s)} - {_fmt(e)} 说话人{spk}\n")
            f.write(text + "\n\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pyannote Diarization + 讯飞 LFASR 转写"
    )
    # 音频
    p.add_argument("--input",  default="托盘贸易-AI初试题.mp3", help="输入音频文件（mp3/wav/m4a/…）")
    p.add_argument("--output", default="result.txt", help="输出 txt 路径")

    # Pyannote
    p.add_argument("--hf-token", dest="hf_token",
                   default=os.environ.get("HF_TOKEN", ""),
                   help="HuggingFace Token（或设环境变量 HF_TOKEN）")
    p.add_argument("--num-speakers", dest="num_speakers", type=int, default=None,
                   help="说话人数量（双人播客填 2，留空则自动估计）")
    p.add_argument("--gap-threshold", dest="gap_threshold", type=float, default=0.8,
                   help="Pyannote 片段合并：同说话人最大间隔秒数（默认 0.8）")
    p.add_argument("--min-duration", dest="min_duration", type=float, default=1.5,
                   help="合并后片段最小秒数，短于此值并入邻段（默认 1.5）")

    # 讯飞
    p.add_argument("--appid",      default=os.environ.get("XFYUN_APPID", ""),
                   help="讯飞 APPID（或设环境变量 XFYUN_APPID）")
    p.add_argument("--secret-key", dest="secret_key",
                   default=os.environ.get("XFYUN_SECRET_KEY", ""),
                   help="LFASR secretkey（或设环境变量 XFYUN_SECRET_KEY）")

    # 并发 & 轮询
    p.add_argument("--workers",       type=int, default=3,
                   help="并发 LFASR 上传线程数（默认 3，不宜过高）")
    p.add_argument("--poll-interval", dest="poll_interval", type=int, default=8)
    p.add_argument("--max-wait",      dest="max_wait",      type=int, default=3600)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.hf_token:
        sys.exit(
            "❌ 未提供 HuggingFace Token\n"
            "   export HF_TOKEN=hf_xxx  或  --hf-token hf_xxx\n"
            "   申请地址：https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    if not args.appid or not args.secret_key:
        sys.exit(
            "❌ 缺少讯飞凭证\n"
            "   export XFYUN_APPID=xxx XFYUN_SECRET_KEY=xxx\n"
            "   或 --appid xxx --secret-key xxx\n"
            "   申请地址：https://www.xfyun.cn/service/lfasr"
        )

    audio_path  = Path(args.input).expanduser().resolve()
    output_file = Path(args.output).expanduser().resolve()

    if not audio_path.exists():
        sys.exit(f"❌ 找不到文件：{audio_path}")
    # ================== 新增代码：音频预处理清洗 ==================
    print("\n[预处理] 正在将音频转换为 16kHz 单声道 WAV 标准格式，避免解码报错...")
    AudioSegment = _import_pydub()
    # 读入原始音频并重采样
    clean_audio = AudioSegment.from_file(str(audio_path))
    clean_audio = clean_audio.set_frame_rate(16000).set_channels(1)
    
    # 存为一个新的 safe.wav 文件
    safe_audio_path = audio_path.with_name(audio_path.stem + "_safe.wav")
    clean_audio.export(str(safe_audio_path), format="wav")
    
    # 将后续流程的输入文件悄悄替换成这个干净的 wav
    audio_path = safe_audio_path 
    print(f"  [预处理] 洗版完成，已生成标准音频: {audio_path.name}")
    # ==============================================================

    # ── Stage 1: Pyannote Diarization ───────────────────────────────────────
    print("\n══ Stage 1 / 3  Pyannote Speaker Diarization ══")
    raw_segments = run_diarization(
        audio_path   = audio_path,
        hf_token     = args.hf_token,
        num_speakers = args.num_speakers,
    )
    print(f"  Pyannote 原始片段数：{len(raw_segments)}")

    merged_segs = merge_diarization(
        raw_segments,
        gap_threshold_s = args.gap_threshold,
        min_duration_s  = args.min_duration,
    )
    labeled_segs, label_map = relabel_speakers(merged_segs)
    print(f"  合并后片段数：{len(labeled_segs)}")
    print(f"  说话人映射：{label_map}")

    # ── Stage 2: 音频切割 ───────────────────────────────────────────────────
    print("\n══ Stage 2 / 3  音频切割 ══")
    with tempfile.TemporaryDirectory(prefix="pyannote_asr_") as tmp:
        tmp_dir = Path(tmp)
        slices  = slice_audio(audio_path, labeled_segs, tmp_dir)

        # ── Stage 3: 讯飞 LFASR ─────────────────────────────────────────────
        print(f"\n══ Stage 3 / 3  讯飞 LFASR（{len(slices)} 段，并发 {args.workers}）══")
        results = transcribe_all(
            appid         = args.appid,
            secret_key    = args.secret_key,
            slices        = slices,
            workers       = args.workers,
            poll_interval = args.poll_interval,
            max_wait      = args.max_wait,
        )
    # tmp_dir 在此自动删除

    # ── 输出 ─────────────────────────────────────────────────────────────────
    non_empty = [(s, e, spk, t) for s, e, spk, t in results if t]
    write_output(non_empty, output_file)

    print(f"\n✅ 完成！输出：{output_file}")
    print("─── 预览 ───────────────────────────────────────────────")
    with output_file.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(line, end="")
            if i >= 30:
                print("  …（更多内容见文件）")
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        sys.exit(1)