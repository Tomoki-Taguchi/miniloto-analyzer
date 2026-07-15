#!/usr/bin/env python3
"""ミニロト データスクレイピング - m-shokai.jp から全抽選データを取得"""

import json
import re
import time
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://m-shokai.jp/miniloto-site/history?page={}"
OUTPUT_PATH = Path(__file__).parent.parent / "docs" / "data" / "miniloto_data.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MiniLotoAnalyzer/1.0)"
}
REQUEST_DELAY = 1.5  # seconds between requests


def load_existing_data():
    """既存データを読み込む。なければ空データを返す。"""
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_updated": None, "total_draws": 0, "draws": []}


def parse_page(html: str) -> list[dict]:
    """1ページ分のHTMLから抽選データをパースする。"""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    draws = []

    rows = soup.select("table tr")
    if not rows:
        # テーブル構造が異なる場合のフォールバック
        rows = soup.find_all("tr")

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        # 回数と日付を取得
        first_cell_text = cells[0].get_text(separator=" ", strip=True)
        round_match = re.search(r"第(\d+)回", first_cell_text)
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", first_cell_text)

        if not round_match or not date_match:
            # 日付形式がYYYY/MM/DD の場合
            date_match = re.search(r"(\d{4}/\d{2}/\d{2})", first_cell_text)
            if not round_match:
                continue

        round_num = int(round_match.group(1))
        if date_match:
            date_str = date_match.group(1).replace("/", "-")
        else:
            date_str = "unknown"

        # 数字を取得 - spanタグから（ミニロトは本数字5個+ボーナス1個の計6個）
        number_spans = cells[1].find_all("span")
        if number_spans:
            numbers = []
            for span in number_spans:
                text = span.get_text(strip=True)
                if text.isdigit():
                    numbers.append(int(text))
            if len(numbers) >= 6:
                main_numbers = sorted(numbers[:5])
                bonus = numbers[5]
            else:
                # spanがうまく取れない場合、テキストから取得
                all_text = cells[1].get_text(separator=" ", strip=True)
                nums = [int(x) for x in re.findall(r"\d+", all_text)]
                if len(nums) >= 6:
                    main_numbers = sorted(nums[:5])
                    bonus = nums[5]
                else:
                    continue
        else:
            # spanがない場合、テキストから直接取得
            all_text = cells[1].get_text(separator=" ", strip=True)
            nums = [int(x) for x in re.findall(r"\d+", all_text)]
            if len(nums) >= 6:
                main_numbers = sorted(nums[:5])
                bonus = nums[5]
            else:
                continue

        # バリデーション
        if len(main_numbers) != 5:
            continue
        if not all(1 <= n <= 31 for n in main_numbers):
            continue
        if not (1 <= bonus <= 31):
            continue

        draws.append({
            "round": round_num,
            "date": date_str,
            "numbers": main_numbers,
            "bonus": bonus
        })

    return draws


def _fetch_page(page: int) -> Optional[list]:
    """1ページ取得してパース結果を返す。失敗時はNone。"""
    url = BASE_URL.format(page)
    print(f"Fetching page {page}: {url}")
    for attempt in range(2):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return parse_page(response.text)
        except requests.RequestException as e:
            print(f"  {'Retry' if attempt else 'Error'} fetching page {page}: {e}")
            if attempt == 0:
                time.sleep(3)
    return None


def _find_last_page() -> int:
    """最終ページ番号を二分探索で特定する。"""
    # まず上限を見つける
    lo, hi = 1, 30
    while hi <= 100:
        draws = _fetch_page(hi)
        time.sleep(REQUEST_DELAY)
        if not draws:
            break
        hi *= 2
    # 二分探索
    lo = hi // 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        draws = _fetch_page(mid)
        time.sleep(REQUEST_DELAY)
        if draws:
            lo = mid
        else:
            hi = mid - 1
    print(f"  Last page: {lo}")
    return lo


def fetch_all_data(existing_last_round: int = 0):
    """全ページからデータを取得する。差分更新対応（最新ページから逆順）。"""
    all_draws = []

    if existing_last_round > 0:
        # 差分更新: 最終ページから逆順に取得し、既存データに追いついたら停止
        last_page = _find_last_page()
        page = last_page
        while page >= 1:
            draws = _fetch_page(page)
            if not draws:
                break

            print(f"  Found {len(draws)} draws (rounds {draws[0]['round']}-{draws[-1]['round']})")
            all_draws.extend(draws)

            oldest_on_page = min(d["round"] for d in draws)
            if oldest_on_page <= existing_last_round:
                print(f"  Reached existing data. Stopping.")
                break

            page -= 1
            time.sleep(REQUEST_DELAY)
    else:
        # 初回: ページ1から順に全取得
        page = 1
        max_pages = 30
        while page <= max_pages:
            draws = _fetch_page(page)
            if not draws:
                print(f"  No draws found on page {page}. Stopping.")
                break

            print(f"  Found {len(draws)} draws (rounds {draws[0]['round']}-{draws[-1]['round']})")
            all_draws.extend(draws)

            page += 1
            time.sleep(REQUEST_DELAY)

    return all_draws


def main():
    print("=== MiniLoto Data Fetcher ===")
    existing = load_existing_data()
    existing_rounds = {d["round"] for d in existing["draws"]}
    last_round = max(existing_rounds) if existing_rounds else 0
    print(f"Existing data: {len(existing_rounds)} draws, last round: {last_round}")

    new_draws = fetch_all_data(last_round)
    print(f"\nFetched {len(new_draws)} draws total")

    # 既存データとマージ（重複排除）
    merged = {d["round"]: d for d in existing["draws"]}
    new_count = 0
    for d in new_draws:
        if d["round"] not in merged:
            new_count += 1
        merged[d["round"]] = d

    all_draws = sorted(merged.values(), key=lambda x: x["round"])

    # 欠番検出: 取得元（第三者サイト）の掲載漏れ等で回号が飛ぶことがあるため警告する。
    # 既存データに手動でバックフィルした回はマージで保持されるので、ここでは検知のみ行う。
    if all_draws:
        present = {d["round"] for d in all_draws}
        rounds = sorted(present)
        gaps = [r for r in range(rounds[0], rounds[-1] + 1) if r not in present]
        if gaps:
            print(f"⚠️  欠番を検出: 第{rounds[0]}回〜第{rounds[-1]}回のうち {len(gaps)} 回が欠落: {gaps}")
            print("    → 取得元に未掲載の可能性があります。別ソースで確認し手動バックフィルを検討してください。")
        else:
            print(f"✓ 回号は連続しています（第{rounds[0]}回〜第{rounds[-1]}回、{len(all_draws)}件）")

    output = {
        "last_updated": datetime.now().isoformat(),
        "total_draws": len(all_draws),
        "draws": all_draws
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_draws)} draws to {OUTPUT_PATH}")
    print(f"New draws added: {new_count}")


if __name__ == "__main__":
    main()
