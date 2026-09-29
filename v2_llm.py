#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
讯飞录音文件转写（大模型版）+ 角色分离  v3
==========================================
接口文档：https://www.xfyun.cn/doc/spark/asr_llm/Ifasr_llm.html
接口地址：https://office-api-ist-dx.iflyaisol.com

鉴权方式（与标准版完全不同）：
  ① 收集所有 URL 查询参数（不含 signature 自身）
  ② 按参数名字典序（TreeMap 规则）排序
  ③ 对每个 value 做 URL 编码，拼成 baseString：
        key1=encoded_val1&key2=encoded_val2&...
  ④ 以 APISecret 为 key 对 baseString 做 HMAC-SHA1
  ⑤ Base64 编码后放入 HTTP 请求头 "signature"

依赖：
    pip install requests

示例：
    python v3.py --input "录音.mp3" --output result.txt --role-num 2
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import random
import string
import sys
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests


# ─── 接口常量 ───────────────────────────────────────────────────────────────
IST_HOST   = "https://office-api-ist-dx.iflyaisol.com"
UPLOAD_API = "/v2/upload"
GET_API    = "/v2/getResult"

# 凭证从环境变量读取，请勿硬编码提交到仓库
DEFAULT_APPID      = os.environ.get("XFYUN_APPID", "")
DEFAULT_API_KEY    = os.environ.get("XFYUN_API_KEY", "")
DEFAULT_API_SECRET = os.environ.get("XFYUN_API_SECRET", "")

ERR_CODE_HINT: Dict[str, str] = {
    "26600": "转写业务通用错误，检查请求参数",
    "26601": "非法应用信息 —— APPID 错误，或未添加「录音文件转写」服务",
    "26602": "任务 ID 不存在",
    "26603": "接口访问频率受限",
    "26604": "获取结果次数超过限制（单订单 100 次上限）",
    "26605": "任务正在处理中",
    "26606": "空音频",
    "26607": "转写语种未授权或已过期",
    "26610": "请求参数错误（签名 / accessKeyId 有误）",
    "26621": "音频文件大小超限（500MB 上限）",
    "26622": "音频时长超限（5 小时上限）",
    "26623": "音频格式不支持",
    "26625": "可用服务时长不足",
    "26633": "可用服务时长不足",
    "26643": "计算音频时长失败，请检查音频是否加密或损坏",
    "26650": "音频格式转换失败，请检查音频是否加密或损坏",
}


# ─── 大模型版鉴权（HMAC-SHA1，对排序后 URL 参数签名）───────────────────────
def _now_datetime_str() -> str:
    tz_8 = timezone(timedelta(hours=8))
    return datetime.now(tz_8).strftime("%Y-%m-%dT%H:%M:%S+0800")


def _random_str(n: int = 16) -> str:
    pool = string.ascii_letters + string.digits
    return "".join(random.choices(pool, k=n))


def make_signature(api_secret: str, params: Dict[str, str]) -> str:
    """
    1. 剔除 'signature'，过滤空值
    2. 按 key 字典序排序
    3. 对 value URL 编码（safe=''）
    4. 拼 baseString: key=encoded_val&...
    5. HMAC-SHA1(api_secret, baseString)，Base64
    """
    sorted_items = sorted(
        (k, v) for k, v in params.items() if k != "signature" and v
    )
    base_string = "&".join(
        f"{k}={quote(str(v), safe='')}" for k, v in sorted_items
    )
    digest = hmac.new(
        api_secret.encode("utf-8"),
        base_string.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def _raise_if_err(body: dict, stage: str) -> None:
    code = str(body.get("code", ""))
    if code == "000000":
        return
    hint = ERR_CODE_HINT.get(code, "")
    raise RuntimeError(
        f"{stage} 失败  code={code}  descInfo={body.get('descInfo')}\n"
        + (f"含义：{hint}\n" if hint else "")
        + f"完整返回：{json.dumps(body, ensure_ascii=False)}"
    )


# ─── 上传 ───────────────────────────────────────────────────────────────────
def upload_file(
    appid: str, api_key: str, api_secret: str,
    file_path: Path,
    role_type: int, role_num: int,
    language: str, pd: Optional[str],
) -> str:
    file_size = file_path.stat().st_size

    params: Dict[str, str] = {
        "appId":                appid,
        "accessKeyId":          api_key,
        "dateTime":             _now_datetime_str(),
        "signatureRandom":      _random_str(16),
        "fileName":             file_path.name,
        "fileSize":             str(file_size),
        "durationCheckDisable": "true",
        "language":             language,
        "roleType":             str(role_type),
        "roleNum":              str(role_num),
    }
    if pd:
        params["pd"] = pd

    sig = make_signature(api_secret, params)
    headers = {
        "Content-Type": "application/octet-stream",
        "signature":    sig,
    }

    url = IST_HOST + UPLOAD_API
    print(f"  → 上传 {file_path.name} ({file_size / 1024 / 1024:.2f} MB)...")
    with file_path.open("rb") as f:
        resp = requests.post(url, params=params, headers=headers,
                             data=f, timeout=600)
    resp.raise_for_status()
    body = resp.json()
    _raise_if_err(body, "上传")

    order_id = body["content"]["orderId"]
    est_ms   = body["content"].get("taskEstimateTime", 0)
    print(f"  ← orderId={order_id}，服务端预估 {est_ms / 1000:.1f} 秒")
    return order_id


# ─── 轮询 ───────────────────────────────────────────────────────────────────
def query_result(
    appid: str, api_key: str, api_secret: str, order_id: str,
) -> dict:
    params: Dict[str, str] = {
        "appId":           appid,
        "accessKeyId":     api_key,
        "dateTime":        _now_datetime_str(),
        "signatureRandom": _random_str(16),
        "orderId":         order_id,
        "resultType":      "transfer",
    }
    sig = make_signature(api_secret, params)
    headers = {"signature": sig}

    resp = requests.post(
        IST_HOST + GET_API, params=params, headers=headers, timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def wait_until_done(
    appid: str, api_key: str, api_secret: str,
    order_id: str, poll_interval: int, max_wait: int,
) -> dict:
    deadline = time.time() + max_wait
    n = 0
    while time.time() < deadline:
        n += 1
        body = query_result(appid, api_key, api_secret, order_id)
        _raise_if_err(body, "查询")
        info   = body["content"]["orderInfo"]
        status = info.get("status")
        if status == 4:
            return body
        if status == -1:
            raise RuntimeError(f"转写失败，failType={info.get('failType')}")
        tag = {0: "已创建", 3: "处理中"}.get(status, f"未知({status})")
        print(f"  · [{n:>3}] status={status} {tag}，{poll_interval}s 后重试...")
        time.sleep(poll_interval)
    raise RuntimeError(f"等待超时：{max_wait}s 内未完成")


# ─── 解析（兼容 lattice2 / nbest 两种结构）──────────────────────────────────
def _extract_text_from_st(st: dict) -> str:
    tokens: List[str] = []
    for rt in st.get("rt", []):
        for ws in rt.get("ws", []):
            for cw in ws.get("cw", []):
                w = cw.get("w", "")
                if w:
                    tokens.append(w)
    for w_item in st.get("w", []):
        if isinstance(w_item, dict):
            tokens.append(w_item.get("w", ""))
        elif isinstance(w_item, str):
            tokens.append(w_item)
    return "".join(tokens).strip()


def parse_order_result(body: dict) -> List[Tuple[int, int, int, str]]:
    order_result_str = body["content"]["orderResult"]
    order_result     = json.loads(order_result_str)
    segments: List[Tuple[int, int, int, str]] = []

    for item in order_result.get("lattice2", []):
        st = item.get("json_1best", {}).get("st", {})
        if not st:
            continue
        try:
            bg = int(st.get("bg", 0))
            ed = int(st.get("ed", 0))
            rl = int(st.get("rl", 0) or 0)
        except (TypeError, ValueError):
            continue
        text = _extract_text_from_st(st)
        if text:
            segments.append((bg, ed, rl, text))

    if not segments:
        for item in order_result.get("nbest", []):
            try:
                bg = int(item.get("bg", 0))
                ed = int(item.get("ed", 0))
                rl = int(item.get("rl", 0) or 0)
            except (TypeError, ValueError):
                continue
            text = item.get("sentence", "").strip() or _extract_text_from_st(item)
            if text:
                segments.append((bg, ed, rl, text))

    segments.sort(key=lambda x: x[0])
    return segments


def merge_segments(
    segments: List[Tuple[int, int, int, str]],
    gap_threshold_ms: int = 2000,
) -> List[Tuple[int, int, int, str]]:
    if not segments:
        return []
    merged = []
    cur_bg, cur_ed, cur_rl, cur_text = segments[0]
    for bg, ed, rl, text in segments[1:]:
        if rl == cur_rl and (bg - cur_ed) < gap_threshold_ms:
            cur_text += text
            cur_ed    = ed
        else:
            merged.append((cur_bg, cur_ed, cur_rl, cur_text))
            cur_bg, cur_ed, cur_rl, cur_text = bg, ed, rl, text
    merged.append((cur_bg, cur_ed, cur_rl, cur_text))
    return merged


# ─── 边界发语词修正 ──────────────────────────────────────────────────────────
_INITIAL_FILLERS: Tuple[str, ...] = (
    "唉咱们", "哎咱们", "那咱们", "嗯咱们", "欸咱们",
    "咱们", "好的", "那个", "这个", "对对", "好吧", "行吧", "嗯嗯",
    "好嘞", "来来", "对吧", "是的", "没错",
    "唉", "哎", "嗯", "啊", "哦", "呃", "诶", "欸", "那", "好", "对",
)
_INTERNAL_PUNCT = "，、,"
_TERMINAL_PUNCT = "。！？?!."


def _peel_trailing_filler(text: str) -> Tuple[str, str]:
    tail_punct = ""
    core = text
    while core and core[-1] in _TERMINAL_PUNCT:
        tail_punct = core[-1] + tail_punct
        core = core[:-1]
    for filler in _INITIAL_FILLERS:
        if not core.endswith(filler):
            continue
        before = core[:-len(filler)]
        if not before or before[-1] not in (_INTERNAL_PUNCT + _TERMINAL_PUNCT):
            continue
        main_body = before.rstrip(_INTERNAL_PUNCT + _TERMINAL_PUNCT + " \t")
        if len(main_body) < 3:
            continue
        return main_body + (tail_punct if tail_punct else "。"), filler
    return text, ""


def fix_boundary_fillers(
    segments: List[Tuple[int, int, int, str]],
    max_gap_ms: int = 2000,
    verbose: bool = False,
) -> List[Tuple[int, int, int, str]]:
    if len(segments) < 2:
        return list(segments)
    result    = [list(s) for s in segments]
    fix_count = 0
    for i in range(len(result) - 1):
        cur_bg, cur_ed, cur_rl, cur_text = result[i]
        nxt_bg, nxt_ed, nxt_rl, nxt_text = result[i + 1]
        if cur_rl == nxt_rl or nxt_bg - cur_ed > max_gap_ms:
            continue
        main, filler = _peel_trailing_filler(cur_text)
        if filler:
            result[i][3]     = main
            result[i + 1][3] = filler + nxt_text
            fix_count += 1
            if verbose:
                print(f"  · 修正 #{fix_count}: 「{filler}」从说话人{cur_rl+1}→说话人{nxt_rl+1}")
    if fix_count and not verbose:
        print(f"  · 修正 {fix_count} 处边界发语词（加 --verbose 可查看明细）")
    return [tuple(s) for s in result]


# ─── 输出 ────────────────────────────────────────────────────────────────────
def format_ts(ms: int) -> str:
    total = max(0, int(ms))
    h, rem = divmod(total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s      = rem // 1_000
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_output(segments, output_file: Path) -> None:
    with output_file.open("w", encoding="utf-8") as f:
        for bg, ed, rl, text in segments:
            f.write(f"{format_ts(bg)} - {format_ts(ed)} 说话人{rl + 1}\n")
            f.write(text + "\n\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="讯飞录音文件转写（大模型版）v3")
    p.add_argument("--input",  default="/Work/ljc/newhope/托盘贸易-AI初试题.mp3")
    p.add_argument("--output", default="result.txt")
    p.add_argument("--appid",      default=DEFAULT_APPID,
                   help="讯飞 APPID（或设环境变量 XFYUN_APPID）")
    p.add_argument("--api-key",    dest="api_key",    default=DEFAULT_API_KEY,
                   help="大模型版 accessKeyId（或设环境变量 XFYUN_API_KEY）")
    p.add_argument("--api-secret", dest="api_secret", default=DEFAULT_API_SECRET,
                   help="大模型版 apiSecret（或设环境变量 XFYUN_API_SECRET）")
    p.add_argument("--language",  default="autodialect",
                   choices=["autodialect", "autominor"])
    p.add_argument("--role-type", dest="role_type", type=int, default=1)
    p.add_argument("--role-num",  dest="role_num",  type=int, default=2)
    p.add_argument("--pd",        default=None)
    p.add_argument("--poll-interval", dest="poll_interval", type=int, default=10)
    p.add_argument("--max-wait",      dest="max_wait",      type=int, default=7200)
    p.add_argument("--no-fix-fillers", dest="fix_fillers",
                   action="store_false", default=True)
    p.add_argument("--dump-raw", dest="dump_raw", default=None)
    p.add_argument("--verbose",  dest="verbose",  action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.appid or not args.api_key or not args.api_secret:
        raise SystemExit(
            "❌ 缺少讯飞凭证\n"
            "   方式一：export XFYUN_APPID=xxx XFYUN_API_KEY=xxx "
            "XFYUN_API_SECRET=xxx\n"
            "   方式二：--appid xxx --api-key xxx --api-secret xxx\n"
            "   申请地址：https://www.xfyun.cn/service/lfasr"
        )

    input_file  = Path(args.input).expanduser().resolve()
    output_file = Path(args.output).expanduser().resolve()

    if not input_file.exists():
        raise SystemExit(f"找不到文件：{input_file}")
    if input_file.stat().st_size / 1024 / 1024 > 500:
        raise SystemExit("文件超过 500MB 上限")

    print(f"[1/3] 上传音频到讯飞 IST 大模型版"
          f"（roleType={args.role_type}, roleNum={args.role_num}, "
          f"language={args.language}）...")
    order_id = upload_file(
        appid=args.appid, api_key=args.api_key, api_secret=args.api_secret,
        file_path=input_file, role_type=args.role_type, role_num=args.role_num,
        language=args.language, pd=args.pd,
    )

    print(f"\n[2/3] 轮询转写进度（每 {args.poll_interval}s）...")
    body = wait_until_done(
        appid=args.appid, api_key=args.api_key, api_secret=args.api_secret,
        order_id=order_id, poll_interval=args.poll_interval, max_wait=args.max_wait,
    )
    real = body["content"]["orderInfo"].get("originalDuration", 0) / 1000.0
    print(f"  ✅ 转写完成，音频真实时长 {real:.1f} 秒\n")

    if args.dump_raw:
        raw_path = Path(args.dump_raw).expanduser().resolve()
        try:
            parsed = json.loads(body["content"]["orderResult"])
            raw_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  📄 原始 orderResult 已保存到 {raw_path}")
        except Exception as e:
            print(f"  ⚠️ 保存原始数据失败: {e}")

    print("[3/3] 解析结果并写文件...")
    raw_segs = parse_order_result(body)
    if not raw_segs:
        print("  ⚠️ 结果为空，原始返回：")
        print(json.dumps(body, ensure_ascii=False, indent=2)[:3000])
        raise SystemExit(1)

    merged = merge_segments(raw_segs)
    if args.fix_fillers:
        merged = fix_boundary_fillers(merged, verbose=args.verbose)

    write_output(merged, output_file)

    counts = Counter(s[2] for s in merged)
    print(f"  共 {len(merged)} 段，{len(counts)} 位说话人：")
    for spk, cnt in sorted(counts.items()):
        print(f"    说话人{spk + 1}: {cnt} 段")

    print(f"\n✅ 输出：{output_file}\n")
    print("─── 预览 ───────────────────────────────────────")
    with output_file.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(line, end="")
            if i > 20:
                print("  ...(更多内容见文件)")
                break


# ─── 单元测试 ────────────────────────────────────────────────────────────────
def _selftest() -> None:
    # 签名排序验证
    sig = make_signature("key", {"signatureRandom": "r", "appId": "a", "dateTime": "d"})
    expected_base = "appId=a&dateTime=d&signatureRandom=r"
    import hmac as _h, hashlib as _hs
    expected = base64.b64encode(
        _h.new(b"key", expected_base.encode(), _hs.sha1).digest()
    ).decode()
    assert sig == expected, f"{sig!r} != {expected!r}"
    print("signature sort OK:", sig)

    _, filler = _peel_trailing_filler(
        "听起来很有意思啊，那我们就直接开始吧，先剖析一下这个复杂的生态，唉咱们."
    )
    assert filler == "唉咱们", filler
    print("filler OK")
    print("\nself-test PASSED ✅")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
        sys.exit(0)
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        sys.exit(1)