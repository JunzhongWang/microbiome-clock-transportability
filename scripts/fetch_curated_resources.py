#!/usr/bin/env python3
"""
fetch_curated_resources.py — 从 Bioconductor BiocHub API 下载 curatedMetagenomicData
的 relative_abundance 资源（零下载路线的取数步骤）。

背景：ExperimentHub 数据现已托管在 https://experimenthub.bioconductor.org（BiocHub API），
真实文件在 location_prefix（XSEDE 对象存储）。旧 Azure 域名已废弃（NXDOMAIN）。

用法：
  python3 scripts/fetch_curated_resources.py                 # 全部 relative_abundance 资源
  python3 scripts/fetch_curated_resources.py QinN_2014 LoombaR_2017   # 只下指定研究
  python3 scripts/fetch_curated_resources.py --liver-only     # 只下肝病研究

特性：幂等续传（已存在且 >1KB 的文件跳过）；每个研究取 rdatadateadded 最新的版本。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

API = "https://experimenthub.bioconductor.org"
OUT = Path(__file__).resolve().parents[1] / "data" / "metaphlan" / "rda"
LISTING_CACHE = Path(__file__).resolve().parents[1] / "data" / "metaphlan" / "hub_resource_list.html"

LIVER_PAT = re.compile(r"cirrhos|NAFLD|NASH|hepat|fibros|steato|hepatocellular", re.I)

UA = {"User-Agent": "Mozilla/5.0 (research script; curatedMetagenomicData fetch)"}


def http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def strip_html_cells(row_html: str) -> list[str]:
    return [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.S)]


def fetch_listing() -> list[dict]:
    """拉取 /package/curatedMetagenomicData 资源表，缓存到本地。"""
    if LISTING_CACHE.exists() and LISTING_CACHE.stat().st_size > 100_000:
        html_text = LISTING_CACHE.read_text()
        print(f"[缓存] 使用已保存的资源清单 {LISTING_CACHE.name}")
    else:
        print("[下载] 拉取 /package/curatedMetagenomicData 资源清单 ...")
        html_text = http_get(f"{API}/package/curatedMetagenomicData", timeout=120).decode("utf-8", "replace")
        LISTING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        LISTING_CACHE.write_text(html_text)
    rows = re.findall(r"<tr>(.*?)</tr>", html_text, re.S)
    recs = [strip_html_cells(r) for r in rows]
    recs = [r for r in recs if r]
    # 字段: [eh_id, ?, ?, title, description]
    out = []
    for r in recs:
        if len(r) >= 4 and r[3].endswith(".relative_abundance"):
            out.append({"eh_id": r[0], "title": r[3]})
    return out


def title_to_study(title: str) -> str:
    """'2021-03-31.QinN_2014.relative_abundance' -> 'QinN_2014'"""
    m = re.match(r"(\d{4}-\d{2}-\d{2})\.(.+)\.relative_abundance$", title)
    return m.group(2) if m else title


def fetch_record(eh_id: str) -> dict | None:
    """/ehid/<id> -> JSON 记录（含 rdatapaths 与 location_prefix）。"""
    for attempt in (1, 2, 3):
        try:
            return json.loads(http_get(f"{API}/ehid/{eh_id[2:] if eh_id.upper().startswith('EH') else eh_id}", timeout=60))
        except Exception as e:
            if attempt == 3:
                print(f"  [失败] {eh_id}: {e}")
                return None
            time.sleep(3 * attempt)


def download_file(url: str, dest: Path, timeout: int = 300) -> bool:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        data = http_get(url, timeout=timeout)
        if len(data) < 500:
            print(f"  [失败] {dest.name}: 文件过小 ({len(data)}B)")
            return False
        tmp.write_bytes(data)
        tmp.rename(dest)
        return True
    except Exception as e:
        print(f"  [失败] {url}: {e}")
        tmp.unlink(missing_ok=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("studies", nargs="*", help="只下载指定研究（如 QinN_2014 LoombaR_2017）")
    ap.add_argument("--liver-only", action="store_true", help="只下载肝病相关研究")
    ap.add_argument("--all", action="store_true", help="下载全部 relative_abundance 资源（约 120 个）")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    listing = fetch_listing()
    print(f"[清单] relative_abundance 资源 {len(listing)} 个")

    # 每个研究取最新日期版本
    by_study: dict[str, list[dict]] = {}
    for r in listing:
        by_study.setdefault(title_to_study(r["title"]), []).append(r)
    chosen = {s: max(v, key=lambda x: x["title"]) for s, v in by_study.items()}
    print(f"[研究] 去重后 {len(chosen)} 个研究")

    if args.studies:
        targets = [s for s in args.studies if s in chosen]
        missing = [s for s in args.studies if s not in chosen]
        if missing:
            print(f"[警告] 清单中找不到: {missing}")
    elif args.all:
        targets = list(chosen)
    else:
        liver = sorted(s for s in chosen if LIVER_PAT.search(s))
        print(f"[提示] 未指定 --all 时默认只下载肝病研究: {liver}")
        print("[提示] 训钟需要健康样本：正式分析请加 --all")
        targets = liver

    print(f"[目标] {len(targets)} 个研究")
    ok, fail = 0, 0
    for i, study in enumerate(sorted(targets), 1):
        rec = chosen[study]
        dest = OUT / f"{study}.rda"
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"({i}/{len(targets)}) [跳过] {study} 已存在")
            ok += 1
            continue
        meta = fetch_record(rec["eh_id"])
        if not meta or not meta.get("rdatapaths"):
            fail += 1
            continue
        prefix = meta["location_prefix"].rstrip("/")
        rd = meta["rdatapaths"][0]["rdatapath"]
        url = f"{prefix}/{rd}"
        t0 = time.time()
        if download_file(url, dest):
            mb = dest.stat().st_size / 1e6
            print(f"({i}/{len(targets)}) [完成] {study}: {mb:.1f} MB ({time.time()-t0:.0f}s)")
            ok += 1
        else:
            fail += 1
        time.sleep(0.3)  # 温和限速

    print(f"\n[汇总] 成功 {ok} / 失败 {fail} → {OUT}")
    if fail:
        print("重跑本脚本可续传失败项")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
