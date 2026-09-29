#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
讯飞录音文件转写（LFASR 标准版）+ 角色分离  v2
==============================================
接口文档：https://www.xfyun.cn/doc/asr/ifasr_new/API.html

相对 v1 的改动：
  [新增] fix_boundary_fillers 后处理 —— 修复讯飞在说话人交接处把下一位的
         发语词（"唉"、"嗯"、"咱们"等）误划到前一位句尾的问题。
  [新增] --dump-raw 选项 —— 把 orderResult 原始 JSON 保存到文件，便于排查。
  [新增] --no-fix-fillers 选项 —— 关闭上述启发式后处理。

依赖：
    pip install requests

示例：
    python xfyun_lfasr_diarization.py \
        --input "录音.mp3" --output result.txt \
        --appid "xxx" --secret-key "xxx" --role-num 2
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


# ─── 接口常量 ───────────────────────────────────────────────────────────────
LFASR_HOST = "https://raasr.xfyun.cn"
UPLOAD_API = "/v2/api/upload"
GET_API    = "/v2/api/getResult"

# ─── 凭证（从环境变量读取，请勿硬编码提交到仓库）────────────────────────────
DEFAULT_APPID      = os.environ.get("XFYUN_APPID", "")
DEFAULT_SECRET_KEY = os.environ.get("XFYUN_SECRET_KEY", "")

ERR_CODE_HINT: Dict[str, str] = {
    "26600": "转写业务通用错误，检查请求参数",
    "26601": "非法应用信息 —— APPID 错误，或未添加「录音文件转写」服务",
    "26602": "任务 ID 不存在",
    "26603": "接口访问频率受限",
    "26604": "获取结果次数超过限制（单订单 100 次上限）",
    "26605": "任务正在处理中",
    "26606": "空音频",
    "26607": "转写语种未授权或已过期",
    "26610": "请求参数错误（99% 是 signa / secretkey 错了）",
    "26621": "音频文件大小超限（500MB 上限）",
    "26622": "音频时长超限（5 小时上限）",
    "26623": "音频格式不支持",
    "26625": "可用服务时长不足",
    "26633": "可用服务时长不足",
    "26643": "计算音频时长失败，请检查音频是否加密或损坏",
    "26650": "音频格式转换失败，请检查音频是否加密或损坏",
}


# ─── 鉴权签名 ────────────────────────────────────────────────────────────────
def make_signa(appid: str, secret_key: str, ts: int) -> str:
    """signa = base64(HmacSHA1(MD5(appid+ts), secretkey))"""
    base_string = (appid + str(ts)).encode("utf-8")
    md5_hex     = hashlib.md5(base_string).hexdigest()
    digest      = hmac.new(
        secret_key.encode("utf-8"),
        md5_hex.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def _raise_if_err(body: dict, stage: str) -> None:
    code = str(body.get("code", ""))
    if code == "000000":
        return
    hint = ERR_CODE_HINT.get(code, "")
    raise RuntimeError(
        f"{stage} 失败 code={code} descInfo={body.get('descInfo')}\n"
        f"{('含义：' + hint) if hint else ''}\n"
        f"完整返回：{json.dumps(body, ensure_ascii=False)}"
    )


# ─── 上传 & 轮询 ────────────────────────────────────────────────────────────
def upload_file(
    appid: str, secret_key: str, file_path: Path,
    role_type: int, role_num: int, language: str, pd: Optional[str],
) -> str:
    ts        = int(time.time())
    signa     = make_signa(appid, secret_key, ts)
    file_size = file_path.stat().st_size
    params = {
        "appId":    appid,  "signa":    signa,  "ts": ts,
        "fileName": file_path.name,  "fileSize": file_size,
        "duration": 200,
        "language": language,  "roleType": role_type,  "roleNum": role_num,
    }
    if pd:
        params["pd"] = pd

    print(f"  → 上传 {file_path.name} ({file_size / 1024 / 1024:.2f} MB)...")
    with file_path.open("rb") as f:
        resp = requests.post(
            LFASR_HOST + UPLOAD_API,
            params={k: str(v) for k, v in params.items()},
            headers={"Content-Type": "application/octet-stream"},
            data=f, timeout=600,
        )
    resp.raise_for_status()
    body = resp.json()
    _raise_if_err(body, "上传")
    order_id = body["content"]["orderId"]
    est_ms   = body["content"].get("taskEstimateTime", 0)
    print(f"  ← orderId={order_id}，服务端预估 {est_ms / 1000:.1f} 秒")
    return order_id


def query_result(appid: str, secret_key: str, order_id: str) -> dict:
    ts    = int(time.time())
    signa = make_signa(appid, secret_key, ts)
    params = {
        "appId": appid, "signa": signa, "ts": ts,
        "orderId": order_id, "resultType": "transfer",
    }
    resp = requests.post(LFASR_HOST + GET_API, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def wait_until_done(
    appid: str, secret_key: str, order_id: str,
    poll_interval: int, max_wait: int,
) -> dict:
    deadline = time.time() + max_wait
    n = 0
    while time.time() < deadline:
        n += 1
        body = query_result(appid, secret_key, order_id)
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


# ─── 解析 ────────────────────────────────────────────────────────────────────
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

        tokens: List[str] = []
        for rt in st.get("rt", []):
            for ws in rt.get("ws", []):
                for cw in ws.get("cw", []):
                    w = cw.get("w", "")
                    if w:
                        tokens.append(w)
        text = "".join(tokens).strip()
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


# ─── 边界发语词修正（v2 新增）──────────────────────────────────────────────
#
# 讯飞 LFASR 的 rl（说话人标签）是在「句子」粒度上打的，词级别没有独立标签。
# 当说话人快速交接、中间几乎没停顿时，下一位的发语词常会被错误地归到上一位的
# 句尾。我们靠两条强信号来发现并修正这种错误：
#   (1) 讯飞自己在发语词前下了「，」—— 说明它的断句模型已经识别出这不是连贯的一句话；
#   (2) 紧接着就是一个不同说话人的段 —— 时间上几乎无缝衔接。
# 两个条件同时成立时，就把尾部的发语词搬到下一段开头。
#
# 这是启发式处理，可能漏修也可能误修，可用 --no-fix-fillers 关掉。

# 按"长的优先"排列（贪心最长匹配）。以下都是"说话人开口时很可能说"的发语词/口头禅。
# 有意避开了「然后/就是/所以/但是/可是/因为」这类连接词 —— 它们也可能是句中成分，
# 单凭"前面有逗号"判断不够保守，会误伤。
_INITIAL_FILLERS: Tuple[str, ...] = (
    # 3 字组合
    "唉咱们", "哎咱们", "那咱们", "嗯咱们", "欸咱们",
    # 2 字
    "咱们", "好的", "那个", "这个", "对对", "好吧", "行吧", "嗯嗯",
    "好嘞", "来来", "对吧", "是的", "没错",
    # 1 字纯语气
    "唉", "哎", "嗯", "啊", "哦", "呃", "诶", "欸", "那", "好", "对",
)

_INTERNAL_PUNCT = "，、,"
_TERMINAL_PUNCT = "。！？?!."


def _peel_trailing_filler(text: str) -> Tuple[str, str]:
    """
    若 text 以 "[有意义内容][内部或终止标点][发语词][可选终止标点]" 收尾，
    返回 (主体带合适的收尾标点, 剥离的发语词)。否则返回 (text, '')。
    """
    # 1) 剥掉末尾的终止标点保留备用
    tail_punct = ""
    core = text
    while core and core[-1] in _TERMINAL_PUNCT:
        tail_punct = core[-1] + tail_punct
        core = core[:-1]

    # 2) 贪心最长匹配发语词
    for filler in _INITIAL_FILLERS:
        if not core.endswith(filler):
            continue
        before = core[:-len(filler)]

        # 关键：发语词必须被"显式标点"隔开，否则是普通句内词，不能剥
        if not before or before[-1] not in (_INTERNAL_PUNCT + _TERMINAL_PUNCT):
            continue

        # 主体去掉尾部任何标点/空白
        main_body = before.rstrip(_INTERNAL_PUNCT + _TERMINAL_PUNCT + " \t")
        # 主体要有实质内容（至少 3 个字符），避免把"，唉"这种本来就短的全剥光
        if len(main_body) < 3:
            continue

        # 给主体补合适的收尾标点：
        #   原来有终止标点 → 保留；
        #   原来没终止标点 → 用中文句号收尾；
        main = main_body + (tail_punct if tail_punct else "。")
        return main, filler

    return text, ""


def fix_boundary_fillers(
    segments: List[Tuple[int, int, int, str]],
    max_gap_ms: int = 2000,
    verbose: bool = False,
) -> List[Tuple[int, int, int, str]]:
    """对相邻换说话人段做发语词迁移；只在时间几乎无缝衔接时触发。"""
    if len(segments) < 2:
        return list(segments)

    result = [list(s) for s in segments]
    fix_count = 0

    for i in range(len(result) - 1):
        cur_bg, cur_ed, cur_rl, cur_text = result[i]
        nxt_bg, nxt_ed, nxt_rl, nxt_text = result[i + 1]

        if cur_rl == nxt_rl:
            continue
        if nxt_bg - cur_ed > max_gap_ms:
            continue

        main, filler = _peel_trailing_filler(cur_text)
        if filler:
            result[i][3]     = main
            result[i + 1][3] = filler + nxt_text
            fix_count += 1
            if verbose:
                print(f"  · 修正 #{fix_count}: 把「{filler}」从"
                      f"说话人{cur_rl + 1}句尾移到说话人{nxt_rl + 1}句首")

    if fix_count and not verbose:
        print(f"  · 修正 {fix_count} 处边界发语词（加 --verbose 可查看明细）")

    return [tuple(s) for s in result]


# ─── 输出 ────────────────────────────────────────────────────────────────────
def format_ts(ms: int) -> str:
    total = max(0, int(ms))
    h = total // 3_600_000
    m = (total % 3_600_000) // 60_000
    s = (total % 60_000) // 1_000
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_output(segments, output_file: Path) -> None:
    with output_file.open("w", encoding="utf-8") as f:
        for bg, ed, rl, text in segments:
            f.write(f"{format_ts(bg)} - {format_ts(ed)} 说话人{rl + 1}\n")
            f.write(text + "\n\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="讯飞 LFASR 录音文件转写 + 角色分离 v2")
    p.add_argument("--input",  default="/Work/ljc/newhope/托盘贸易-AI初试题.mp3", help="输入音频（mp3/wav/m4a/flac/...）")
    p.add_argument("--output", default="result.txt")

    p.add_argument("--appid",      default=DEFAULT_APPID,
                   help="讯飞 APPID（或设环境变量 XFYUN_APPID）")
    p.add_argument("--secret-key", dest="secret_key",
                   default=DEFAULT_SECRET_KEY,
                   help=("LFASR 的 secretkey，请从控制台「录音文件转写标准版」"
                         "服务管理页复制。或设环境变量 XFYUN_SECRET_KEY。"
                         "注意 Base64 中 I(大写i) 和 l(小写L) "
                         "视觉几乎一样，务必直接粘贴不要手打。"))

    p.add_argument("--language", default="cn")
    p.add_argument("--role-type", dest="role_type", type=int, default=1)
    p.add_argument("--role-num",  dest="role_num",  type=int, default=0,
                   help="0=盲分；2~10=指定人数（明确人数精度更高）")
    p.add_argument("--pd", default=None,
                   help="领域：finance/ecom/tech/edu/medical/...")

    p.add_argument("--poll-interval", dest="poll_interval", type=int, default=10)
    p.add_argument("--max-wait",      dest="max_wait",      type=int, default=7200)

    # v2 新增
    p.add_argument("--no-fix-fillers", dest="fix_fillers",
                   action="store_false", default=True,
                   help="关闭边界发语词自动迁移")
    p.add_argument("--dump-raw", dest="dump_raw", default=None,
                   help="把讯飞原始 orderResult JSON 保存到此文件，便于排查")
    p.add_argument("--verbose",  dest="verbose",  action="store_true",
                   help="显示详细修正日志")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.appid or not args.secret_key:
        raise SystemExit(
            "❌ 缺少讯飞凭证\n"
            "   方式一：export XFYUN_APPID=xxx XFYUN_SECRET_KEY=xxx\n"
            "   方式二：--appid xxx --secret-key xxx\n"
            "   申请地址：https://www.xfyun.cn/service/lfasr"
        )

    input_file  = Path(args.input).expanduser().resolve()
    output_file = Path(args.output).expanduser().resolve()

    if not input_file.exists():
        raise SystemExit(f"找不到文件：{input_file}")
    size_mb = input_file.stat().st_size / 1024 / 1024
    if size_mb > 500:
        raise SystemExit(f"文件超过 500MB 上限：{size_mb:.1f} MB")

    print(f"[1/3] 上传音频到讯飞 LFASR（roleType={args.role_type}, "
          f"roleNum={args.role_num}, language={args.language}）...")
    order_id = upload_file(
        appid=args.appid, secret_key=args.secret_key, file_path=input_file,
        role_type=args.role_type, role_num=args.role_num,
        language=args.language, pd=args.pd,
    )

    print(f"\n[2/3] 轮询转写进度（每 {args.poll_interval}s）...")
    body = wait_until_done(
        appid=args.appid, secret_key=args.secret_key, order_id=order_id,
        poll_interval=args.poll_interval, max_wait=args.max_wait,
    )
    info = body["content"]["orderInfo"]
    real = info.get("originalDuration", 0) / 1000.0
    print(f"  ✅ 转写完成，音频真实时长 {real:.1f} 秒\n")

    # 原始数据落盘（可选）
    if args.dump_raw:
        raw_path = Path(args.dump_raw).expanduser().resolve()
        try:
            parsed = json.loads(body["content"]["orderResult"])
            raw_path.write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  📄 原始 orderResult 已保存到 {raw_path}")
        except Exception as e:
            print(f"  ⚠️ 保存原始数据失败: {e}")

    print("[3/3] 解析结果并写文件...")
    raw_segs = parse_order_result(body)
    if not raw_segs:
        print("  ⚠️ 结果为空")
        print(json.dumps(body, ensure_ascii=False, indent=2)[:2000])
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


# ─── 小型单元测试（python xxx.py --selftest）────────────────────────────────
def _selftest() -> None:
    # 用户实际遇到的那条
    main, filler = _peel_trailing_filler(
        "听起来很有意思啊，那我们就直接开始吧，先剖析一下这个复杂的生态，唉咱们."
    )
    assert filler == "唉咱们", f"expected 唉咱们, got {filler!r}"
    assert main.endswith("生态.") or main.endswith("生态。"), main

    # 不应误剥：主体太短
    main, filler = _peel_trailing_filler("，唉咱们.")
    assert filler == "", f"主体太短不应剥，却剥出了 {filler!r}"

    # 不应误剥：发语词前无标点（贴着正文）
    main, filler = _peel_trailing_filler("我觉得没问题啊.")
    assert filler == "", f"无前置标点不应剥，却剥出了 {filler!r}"

    # 应剥：用句号分隔
    main, filler = _peel_trailing_filler("我说过了。好的")
    assert filler == "好的", filler

    # 应剥：以单字语气词收尾
    main, filler = _peel_trailing_filler("我们就这样决定了，嗯。")
    assert filler == "嗯", filler

    print("self-test OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
        sys.exit(0)
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        sys.exit(1)