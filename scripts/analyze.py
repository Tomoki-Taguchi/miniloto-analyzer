#!/usr/bin/env python3
"""ミニロト 分析エンジン - 統計分析 + 予想生成（v3: 周期分析・RF・LSTM搭載）"""

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn

DATA_PATH = Path(__file__).parent.parent / "docs" / "data" / "miniloto_data.json"
OUTPUT_PATH = Path(__file__).parent.parent / "docs" / "data" / "analysis.json"

# 毎日同じ予想を出すためにシードを日付で固定
random.seed(date.today().isoformat())


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 1. 出現頻度分析（改善: 直近重み付き頻度を追加）
# ============================================================
def analyze_frequency(draws):
    counter = Counter()
    last_seen = {}
    current_round = draws[-1]["round"]

    # 全期間
    for d in draws:
        for n in d["numbers"]:
            counter[n] += 1
            last_seen[n] = d["round"]

    total = len(draws)
    counts = {str(n): counter.get(n, 0) for n in range(1, 32)}
    percentages = {str(n): round(counter.get(n, 0) / total * 100, 2) for n in range(1, 32)}
    drought = {str(n): current_round - last_seen.get(n, 0) for n in range(1, 32)}

    # 直近100回・300回の出現頻度（改善②）
    recent_100 = Counter()
    recent_300 = Counter()
    for d in draws[-100:]:
        for n in d["numbers"]:
            recent_100[n] += 1
    for d in draws[-300:]:
        for n in d["numbers"]:
            recent_300[n] += 1

    recent_100_pct = {str(n): round(recent_100.get(n, 0) / min(100, total) * 100, 2) for n in range(1, 32)}
    recent_300_pct = {str(n): round(recent_300.get(n, 0) / min(300, total) * 100, 2) for n in range(1, 32)}

    # 各数字の平均出現間隔を計算（改善③で使用）
    appearances = defaultdict(list)
    for d in draws:
        for n in d["numbers"]:
            appearances[n].append(d["round"])
    avg_intervals = {}
    for n in range(1, 32):
        rounds = sorted(appearances.get(n, []))
        if len(rounds) >= 2:
            intervals = [rounds[i+1] - rounds[i] for i in range(len(rounds)-1)]
            avg_intervals[str(n)] = round(sum(intervals) / len(intervals), 2)
        else:
            avg_intervals[str(n)] = total  # 出現が少ない場合

    sorted_by_freq = sorted(range(1, 32), key=lambda n: counter.get(n, 0), reverse=True)
    hot = sorted_by_freq[:10]
    cold = sorted_by_freq[-10:]

    return {
        "counts": counts,
        "percentages": percentages,
        "drought": drought,
        "hot": hot,
        "cold": cold,
        "recent_100": recent_100_pct,
        "recent_300": recent_300_pct,
        "avg_intervals": avg_intervals,
    }


# ============================================================
# 2. 引っ張り分析
# ============================================================
def analyze_pull(draws):
    distribution = Counter()
    pull_details = []

    for i in range(1, len(draws)):
        prev = set(draws[i - 1]["numbers"])
        curr = set(draws[i]["numbers"])
        overlap = prev & curr
        distribution[len(overlap)] += 1
        if i >= len(draws) - 20:
            pull_details.append({
                "round": draws[i]["round"],
                "date": draws[i]["date"],
                "numbers": draws[i]["numbers"],
                "pulled": sorted(list(overlap)),
                "pull_count": len(overlap),
            })

    total_transitions = len(draws) - 1
    avg = sum(k * v for k, v in distribution.items()) / total_transitions if total_transitions else 0

    return {
        "distribution": {str(k): v for k, v in sorted(distribution.items())},
        "average": round(avg, 2),
        "last_draw_numbers": draws[-1]["numbers"],
        "recent_pulls": pull_details[-10:],
    }


# ============================================================
# 3. 数字帯分析
# ============================================================
def get_zone(n):
    if n <= 10:
        return "low"
    elif n <= 21:
        return "mid"
    else:
        return "high"


def analyze_zone(draws):
    pattern_counter = Counter()
    zone_totals = {"low": 0, "mid": 0, "high": 0}

    for d in draws:
        zones = [get_zone(n) for n in d["numbers"]]
        low_c = zones.count("low")
        mid_c = zones.count("mid")
        high_c = zones.count("high")
        pattern = f"{low_c}-{mid_c}-{high_c}"
        pattern_counter[pattern] += 1
        zone_totals["low"] += low_c
        zone_totals["mid"] += mid_c
        zone_totals["high"] += high_c

    total = len(draws) * 5
    zone_averages = {k: round(v / total * 100, 2) for k, v in zone_totals.items()}
    top_patterns = [
        {"pattern": p, "count": c, "percentage": round(c / len(draws) * 100, 1)}
        for p, c in pattern_counter.most_common(10)
    ]

    return {
        "zone_averages": zone_averages,
        "top_patterns": top_patterns,
    }


# ============================================================
# 4. ペア分析（改善④: 直近ペアも分析）
# ============================================================
def analyze_pairs(draws):
    pair_counter = Counter()
    recent_pair_counter = Counter()  # 直近200回

    for d in draws:
        for pair in combinations(d["numbers"], 2):
            pair_counter[pair] += 1

    for d in draws[-200:]:
        for pair in combinations(d["numbers"], 2):
            recent_pair_counter[pair] += 1

    top_pairs = [
        {"pair": list(pair), "count": count}
        for pair, count in pair_counter.most_common(30)
    ]

    # 各数字の相性マップ（上位5つ）
    affinity = {}
    for n in range(1, 32):
        partners = [(p, c) for p, c in pair_counter.items() if n in p]
        partners.sort(key=lambda x: x[1], reverse=True)
        top5 = []
        for pair, count in partners[:5]:
            other = pair[1] if pair[0] == n else pair[0]
            top5.append({"number": other, "count": count})
        affinity[str(n)] = top5

    return {
        "top_pairs": top_pairs,
        "affinity": affinity,
        "pair_counts": {f"{a}-{b}": c for (a, b), c in pair_counter.items()},
        "recent_pair_counts": {f"{a}-{b}": c for (a, b), c in recent_pair_counter.items()},
    }


# ============================================================
# 5. 連番分析（改善⑤: 新規追加）
# ============================================================
def analyze_consecutive(draws):
    """隣接数字（連番）の出現傾向を分析"""
    consec_count = 0
    total_draws = len(draws)

    for d in draws:
        nums = sorted(d["numbers"])
        for i in range(len(nums) - 1):
            if nums[i + 1] - nums[i] == 1:
                consec_count += 1
                break  # 1回の抽選で1カウント

    consec_rate = round(consec_count / total_draws * 100, 1)

    # 各数字が連番で出現した回数
    consec_partner_counts = Counter()
    for d in draws:
        nums = sorted(d["numbers"])
        for i in range(len(nums) - 1):
            if nums[i + 1] - nums[i] == 1:
                consec_partner_counts[nums[i]] += 1
                consec_partner_counts[nums[i + 1]] += 1

    return {
        "has_consecutive_rate": consec_rate,
        "partner_counts": {str(n): consec_partner_counts.get(n, 0) for n in range(1, 32)},
    }


# ============================================================
# 6. N回周期分析
# ============================================================
def analyze_cycle(draws):
    """各数字の出現周期を検出し、次回出現の期待度を算出"""
    total_draws = len(draws)
    current_round = draws[-1]["round"]

    cycle_data = {}
    for n in range(1, 32):
        # 出現した回のリスト
        appearances = [d["round"] for d in draws if n in d["numbers"]]
        if len(appearances) < 3:
            cycle_data[str(n)] = {
                "dominant_cycle": None,
                "cycle_score": 0.0,
                "intervals": [],
                "next_expected": None,
            }
            continue

        # 出現間隔を計算
        intervals = [appearances[i + 1] - appearances[i] for i in range(len(appearances) - 1)]
        avg_interval = sum(intervals) / len(intervals)

        # 最頻出の間隔（周期候補）を検出
        interval_counter = Counter(intervals)
        # 近い間隔をグルーピング（±1の範囲）
        grouped = defaultdict(int)
        for iv, cnt in interval_counter.items():
            grouped[iv] += cnt
        # 上位3つの周期候補
        top_cycles = sorted(grouped.items(), key=lambda x: x[1], reverse=True)[:3]
        dominant_cycle = top_cycles[0][0] if top_cycles else int(avg_interval)

        # 最後の出現からの経過
        since_last = current_round - appearances[-1]

        # 周期スコア: 支配的周期に対してどれだけ「次に来そう」か
        # 周期の倍数に近いほどスコアが高い
        if dominant_cycle > 0:
            remainder = since_last % dominant_cycle
            closeness = 1.0 - (min(remainder, dominant_cycle - remainder) / (dominant_cycle / 2))
            # 1周期以上経過でボーナス
            cycle_multiplier = min(since_last / dominant_cycle, 2.0)
            score = closeness * cycle_multiplier
        else:
            score = 0.0

        next_expected = appearances[-1] + dominant_cycle

        cycle_data[str(n)] = {
            "dominant_cycle": dominant_cycle,
            "cycle_score": round(score, 4),
            "avg_interval": round(avg_interval, 1),
            "since_last": since_last,
            "next_expected": next_expected,
            "top_cycles": [{"cycle": c, "count": cnt} for c, cnt in top_cycles[:3]],
        }

    return cycle_data


# ============================================================
# 7. ランダムフォレスト予測
# ============================================================
def build_rf_features(draws, target_idx, n):
    """各数字について、特徴量と教師ラベルを構築"""
    window = 20
    if target_idx < window:
        return None, None

    features = []
    # 直近window回の出現 (0/1)
    for i in range(window):
        features.append(1.0 if n in draws[target_idx - 1 - i]["numbers"] else 0.0)
    # 直近5/10/20回の出現率
    for w in [5, 10, 20]:
        count = sum(1 for i in range(w) if n in draws[target_idx - 1 - i]["numbers"])
        features.append(count / w)
    # 最後に出てからの経過回数
    since_last = 0
    for i in range(target_idx - 1, -1, -1):
        if n in draws[i]["numbers"]:
            since_last = target_idx - 1 - i
            break
    features.append(since_last / 20.0)  # 正規化
    # 前回出たか
    features.append(1.0 if n in draws[target_idx - 1]["numbers"] else 0.0)

    label = 1 if n in draws[target_idx]["numbers"] else 0
    return features, label


def predict_rf(draws):
    """ランダムフォレストで各数字の次回出現確率を予測"""
    print("  Training Random Forest...")
    rf_scores = {}
    window = 20

    for n in range(1, 32):
        X, y = [], []
        for idx in range(window, len(draws)):
            feat, label = build_rf_features(draws, idx, n)
            if feat is not None:
                X.append(feat)
                y.append(label)

        X = np.array(X)
        y = np.array(y)

        if len(set(y)) < 2:
            rf_scores[n] = 0.5
            continue

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X, y)

        # 次回の特徴量を作成して予測
        next_feat, _ = build_rf_features(draws, len(draws) - 1, n)
        if next_feat is None:
            rf_scores[n] = 0.5
            continue

        # 最後のデータで次回を予測するために特徴量を再構築
        feat = []
        for i in range(window):
            feat.append(1.0 if n in draws[len(draws) - 1 - i]["numbers"] else 0.0)
        for w in [5, 10, 20]:
            count = sum(1 for i in range(w) if n in draws[len(draws) - 1 - i]["numbers"])
            feat.append(count / w)
        since_last = 0
        for i in range(len(draws) - 1, -1, -1):
            if n in draws[i]["numbers"]:
                since_last = len(draws) - 1 - i
                break
        feat.append(since_last / 20.0)
        feat.append(1.0 if n in draws[-1]["numbers"] else 0.0)

        prob = clf.predict_proba(np.array([feat]))[0]
        rf_scores[n] = float(prob[1]) if len(prob) > 1 else 0.5

    return rf_scores


# ============================================================
# 8. LSTM予測
# ============================================================
class LottoLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


def predict_lstm(draws, seq_len=30, epochs=50):
    """LSTMで各数字の次回出現確率を予測"""
    print("  Training LSTM...")
    torch.manual_seed(42)
    lstm_scores = {}

    for n in range(1, 32):
        # 時系列データ作成: 各回で出現=1, 未出現=0
        series = np.array([1.0 if n in d["numbers"] else 0.0 for d in draws], dtype=np.float32)

        if len(series) < seq_len + 1:
            lstm_scores[n] = 0.5
            continue

        # スライディングウィンドウでデータセット作成
        X, y = [], []
        for i in range(len(series) - seq_len):
            X.append(series[i:i + seq_len])
            y.append(series[i + seq_len])

        X = torch.tensor(np.array(X)).unsqueeze(-1)  # (samples, seq_len, 1)
        y = torch.tensor(np.array(y)).unsqueeze(-1)  # (samples, 1)

        model = LottoLSTM(input_size=1, hidden_size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.BCELoss()

        # 学習
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # 次回の予測
        model.eval()
        with torch.no_grad():
            last_seq = torch.tensor(series[-seq_len:]).unsqueeze(0).unsqueeze(-1)
            prob = model(last_seq).item()

        lstm_scores[n] = prob

    return lstm_scores


# ============================================================
# 9. AI予想エンジン（v3: 10要素統合）
# ============================================================
# 各モードの基準重み（頻度/干ばつ/引っ張り/ペア/連番/直近/周期/RF/LSTM/ランダム）
_BASE_MODES = {
    "balanced":        {"freq": 0.15, "drought": 0.12, "pull": 0.10, "pair": 0.10, "consec": 0.05, "recent": 0.08, "cycle": 0.12, "rf": 0.12, "lstm": 0.12, "random": 0.04},
    "frequency_heavy": {"freq": 0.30, "drought": 0.08, "pull": 0.07, "pair": 0.07, "consec": 0.03, "recent": 0.12, "cycle": 0.08, "rf": 0.08, "lstm": 0.08, "random": 0.09},
    "pull_heavy":      {"freq": 0.10, "drought": 0.08, "pull": 0.28, "pair": 0.10, "consec": 0.05, "recent": 0.04, "cycle": 0.08, "rf": 0.09, "lstm": 0.09, "random": 0.09},
    "zone_balanced":   {"freq": 0.12, "drought": 0.12, "pull": 0.07, "pair": 0.10, "consec": 0.05, "recent": 0.07, "cycle": 0.12, "rf": 0.10, "lstm": 0.10, "random": 0.15},
    "pair_heavy":      {"freq": 0.10, "drought": 0.08, "pull": 0.07, "pair": 0.22, "consec": 0.05, "recent": 0.07, "cycle": 0.08, "rf": 0.10, "lstm": 0.10, "random": 0.13},
    "ml_heavy":        {"freq": 0.08, "drought": 0.05, "pull": 0.05, "pair": 0.05, "consec": 0.02, "recent": 0.05, "cycle": 0.10, "rf": 0.25, "lstm": 0.25, "random": 0.10},
}

# 多様性チューニング（特定の数字が全予想を独占するのを防ぐ）
DIVERSITY_TOPK = 9          # 貪欲選択の各ステップで揺らす上位候補数の基準
DIVERSITY_QUALITY_W = 0.25  # トライアル選定における「平均スコア」の重み（小さいほど多様）
DIVERSITY_JITTER = 0.0      # トライアル選定に加えるランダム揺らぎ量
DIVERSITY_ECHO_REDUCTION = 0.25  # 相関の高い echo 指標の重複加点を削る割合
# freq/recent/cycle/rf/lstm は「よく出る数字」を重複評価する相関の高い指標群。
# これらが1つの数字を何重にも押し上げ全予想を独占させる原因になるため、
# 各モードの副次的な echo 指標を一定割合だけ削り、削った分を探索(random)に回す。
_ECHO_FACTORS = {"freq", "recent", "cycle", "rf", "lstm"}


def _apply_diversity(modes, d):
    """echo 指標の重複加点を抑え、削った分を探索(random)へ。各モードの最重要指標
    (signature)は保護してモードの個性を維持する。"""
    out = {}
    for mk, w in modes.items():
        w = dict(w)
        sig = max(w, key=w.get)  # そのモードで最重要の指標は削らない
        moved = 0.0
        for f in _ECHO_FACTORS:
            if f == sig:
                continue
            cut = w[f] * d
            w[f] = round(w[f] - cut, 4)
            moved += cut
        w["random"] = round(w["random"] + moved, 4)
        out[mk] = w
    return out


MODES = _apply_diversity(_BASE_MODES, DIVERSITY_ECHO_REDUCTION)

MODE_NAMES = {
    "balanced": "総合予想",
    "frequency_heavy": "出現頻度重視",
    "pull_heavy": "引っ張り重視",
    "zone_balanced": "数字帯バランス重視",
    "pair_heavy": "ペア重視",
    "ml_heavy": "AI(RF+LSTM)重視",
}


def normalize(values: dict) -> dict:
    """0-1に正規化"""
    vals = list(values.values())
    min_v, max_v = min(vals), max(vals)
    rng = max_v - min_v if max_v != min_v else 1
    return {k: (v - min_v) / rng for k, v in values.items()}


# ============================================================
# モンテカルロ・シミュレーション関連
# ============================================================
def _seed_from_str(s: str) -> int:
    """文字列から再現可能な32bit整数シードを生成（日替わりで固定結果にするため）"""
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**32)


def monte_carlo_confidence(scores: dict, n_trials: int = 10000, seed_str: str = "mc") -> dict:
    """各数字のスコアを重みとした非復元抽出をn_trials回シミュレーションし、
    各数字が5個の組に選ばれた割合(%)を「モンテカルロ信頼度」として返す。
    Efraimidis-Spirakis法（重み付きリザーバーサンプリング）でベクトル化。
    """
    numbers = list(range(1, 32))
    weights_arr = np.array([max(scores.get(n, 0.0), 1e-6) for n in numbers])
    rng = np.random.default_rng(_seed_from_str(seed_str))

    u = rng.random((n_trials, len(numbers)))
    keys = u ** (1.0 / weights_arr)
    top5_idx = np.argpartition(-keys, 5, axis=1)[:, :5]

    counts = np.zeros(len(numbers))
    np.add.at(counts, top5_idx.ravel(), 1)
    pct = counts / n_trials * 100

    return {numbers[i]: round(float(pct[i]), 2) for i in range(len(numbers))}


def simulate_random_baseline(total_rounds: int, n_sim: int = 200000, seed_str: str = "baseline"):
    """完全ランダムに5個選んだ場合の成績をモンテカルロ・シミュレーションし、
    AI予想モードの実績と比較するための基準値を作る。
    """
    if total_rounds <= 0:
        return None

    rng = np.random.default_rng(_seed_from_str(seed_str))
    matches = rng.hypergeometric(ngood=5, nbad=26, nsample=5, size=n_sim)
    bonus_hits = rng.random(n_sim) < (1 / 26)

    match_dist_pct = {str(i): round(float((matches == i).mean() * 100), 3) for i in range(6)}
    prize_prob = {
        "1st": float((matches == 5).mean()),
        "2nd": float(((matches == 4) & bonus_hits).mean()),
        "3rd": float(((matches == 4) & ~bonus_hits).mean()),
        "4th": float((matches == 3).mean()),
    }
    prize_expected = {k: round(v * total_rounds, 4) for k, v in prize_prob.items()}

    return {
        "mode_name": "ランダム基準（シミュレーション）",
        "n_simulations": n_sim,
        "total_rounds": total_rounds,
        "avg_matched": round(float(matches.mean()), 3),
        "match_distribution_pct": match_dist_pct,
        "prize_expected": prize_expected,
    }


def generate_prediction(freq_data, pull_data, zone_data, pair_data, consec_data, cycle_data, rf_scores_raw, lstm_scores_raw, draws, weights, mode_key=None, period_label=None):
    """1つの予想を生成（v3: 10要素統合）"""
    total_draws = len(draws)

    # --- スコア計算 ---

    # ① 出現頻度スコア（全期間）
    freq_scores = normalize({n: freq_data["counts"].get(str(n), 0) for n in range(1, 32)})

    # ② 直近重み付き頻度スコア（改善: 直近100回+300回の加重平均）
    recent_scores = {}
    for n in range(1, 32):
        r100 = float(freq_data["recent_100"].get(str(n), 0))
        r300 = float(freq_data["recent_300"].get(str(n), 0))
        # 直近100回を重視（0.6:0.4）
        recent_scores[n] = r100 * 0.6 + r300 * 0.4
    recent_scores = normalize(recent_scores)

    # ③ 干ばつスコア（改善: 平均間隔と比較した相対的な遅延度）
    drought_raw = {}
    for n in range(1, 32):
        current_drought = freq_data["drought"].get(str(n), 0)
        avg_interval = freq_data["avg_intervals"].get(str(n), 7)
        # 平均間隔に対してどれだけ遅延しているか（1.0 = 平均通り、2.0 = 平均の2倍遅い）
        if avg_interval > 0:
            drought_raw[n] = current_drought / avg_interval
        else:
            drought_raw[n] = 0
    drought_scores = normalize(drought_raw)

    # ④ 引っ張りスコア（改善: 直近N回の出現頻度でグラデーション化）
    pull_scores = {}
    recent_5 = [set(d["numbers"]) for d in draws[-5:]]
    for n in range(1, 32):
        # 直近5回での出現回数を加重（直近ほど重い）
        weighted_pull = 0.0
        weights_pull = [0.40, 0.25, 0.15, 0.12, 0.08]  # 最新→古い
        for i, nums in enumerate(reversed(recent_5)):
            if n in nums:
                weighted_pull += weights_pull[i]
        pull_scores[n] = weighted_pull
    pull_scores = normalize(pull_scores)

    # ⑤ 連番スコア
    consec_scores_raw = {n: consec_data["partner_counts"].get(str(n), 0) for n in range(1, 32)}
    consec_base_scores = normalize(consec_scores_raw)

    # ⑥ 周期スコア
    cycle_scores = normalize({n: cycle_data.get(str(n), {}).get("cycle_score", 0) for n in range(1, 32)})

    # ⑦ ランダムフォレストスコア
    rf_scores = normalize({n: rf_scores_raw.get(n, 0.5) for n in range(1, 32)})

    # ⑧ LSTMスコア
    lstm_scores = normalize({n: lstm_scores_raw.get(n, 0.5) for n in range(1, 32)})

    # ⑨ モンテカルロ信頼度スコア（ペア・ランダム要素を除く各スコアの加重和を確率として
    #    重み付き非復元抽出を1万回シミュレーションし、各数字が選ばれた割合を算出）
    base_scores = {
        n: (
            weights["freq"] * freq_scores.get(n, 0)
            + weights["recent"] * recent_scores.get(n, 0)
            + weights["drought"] * drought_scores.get(n, 0)
            + weights["pull"] * pull_scores.get(n, 0)
            + weights["consec"] * consec_base_scores.get(n, 0)
            + weights["cycle"] * cycle_scores.get(n, 0)
            + weights["rf"] * rf_scores.get(n, 0)
            + weights["lstm"] * lstm_scores.get(n, 0)
        )
        for n in range(1, 32)
    }
    mc_seed = f"{date.today().isoformat()}_{mode_key}_{period_label}_mc"
    mc_confidence = monte_carlo_confidence(base_scores, n_trials=10000, seed_str=mc_seed)

    # ペアカウント辞書（全期間+直近を混合）
    pair_counts_all = pair_data.get("pair_counts", {})
    pair_counts_recent = pair_data.get("recent_pair_counts", {})

    # 合計値の制約（改善⑥: 標準偏差ベース ±1σ）
    sums = [sum(d["numbers"]) for d in draws]
    avg_sum = sum(sums) / len(sums)
    std_sum = math.sqrt(sum((s - avg_sum) ** 2 for s in sums) / len(sums))
    sum_range = (avg_sum - std_sum, avg_sum + std_sum)

    last_numbers = set(pull_data["last_draw_numbers"])

    # --- 改善⑦: 複数候補から最良を選ぶ（貪欲法のバイアス軽減） ---
    NUM_TRIALS = 20
    best_result = None
    best_score = -1

    for trial in range(NUM_TRIALS):
        selected = []
        sel_reasons = {}

        for step in range(5):
            candidates = []
            for n in range(1, 32):
                if n in selected:
                    continue

                # ペアスコア: 全期間60% + 直近200回40%
                pair_score = 0.0
                if selected:
                    for s in selected:
                        key = f"{min(n,s)}-{max(n,s)}"
                        all_c = pair_counts_all.get(key, 0)
                        rec_c = pair_counts_recent.get(key, 0) * (total_draws / min(200, total_draws))
                        pair_score += all_c * 0.6 + rec_c * 0.4
                    pair_score /= len(selected)

                # 連番ボーナス: 選出済みの数字の隣にある場合スコアアップ
                consec_bonus = consec_base_scores.get(n, 0)
                if selected:
                    for s in selected:
                        if abs(n - s) == 1:
                            consec_bonus = 1.0
                            break

                candidates.append((n, {
                    "freq": freq_scores.get(n, 0),
                    "recent": recent_scores.get(n, 0),
                    "drought": drought_scores.get(n, 0),
                    "pull": pull_scores.get(n, 0),
                    "pair_raw": pair_score,
                    "consec": consec_bonus,
                    "cycle": cycle_scores.get(n, 0),
                    "rf": rf_scores.get(n, 0),
                    "lstm": lstm_scores.get(n, 0),
                    "random": random.random(),
                }))

            # ペアスコアを正規化
            pair_raws = [c[1]["pair_raw"] for c in candidates]
            pr_min, pr_max = min(pair_raws), max(pair_raws)
            pr_range = pr_max - pr_min if pr_max != pr_min else 1
            for _, scores in candidates:
                scores["pair"] = (scores["pair_raw"] - pr_min) / pr_range

            # 総合スコア計算
            scored = []
            for n, scores in candidates:
                total = (
                    weights["freq"] * scores["freq"]
                    + weights["recent"] * scores["recent"]
                    + weights["drought"] * scores["drought"]
                    + weights["pull"] * scores["pull"]
                    + weights["pair"] * scores["pair"]
                    + weights["consec"] * scores["consec"]
                    + weights["cycle"] * scores["cycle"]
                    + weights["rf"] * scores["rf"]
                    + weights["lstm"] * scores["lstm"]
                    + weights["random"] * scores["random"]
                )
                scored.append((n, total, scores))

            scored.sort(key=lambda x: x[1], reverse=True)

            # トライアルごとにトップNからランダムに揺らす（探索を強化し、特定の数字が
            # 全予想を独占するのを防ぐ）。※旧実装は scored[:top_k] のスライスコピーを
            # シャッフルしており実質無効だったのを修正。
            if len(scored) > DIVERSITY_TOPK:
                top_k = min(DIVERSITY_TOPK + trial, len(scored))
                head = scored[:top_k]
                random.shuffle(head)
                scored[:top_k] = head

            # 制約チェックしながら選出
            found = False
            for n, total_score, scores in scored:
                test_selected = selected + [n]

                # 帯バランスチェック
                zones = [get_zone(x) for x in test_selected]
                zone_counts = Counter(zones)
                remaining = 5 - len(test_selected)

                if any(c > 3 for c in zone_counts.values()):
                    continue
                missing_zones = [z for z in ["low", "mid", "high"] if zone_counts.get(z, 0) == 0]
                if len(missing_zones) > remaining:
                    continue

                # 奇偶バランス（全数字が同じ偶奇に偏るのを回避）
                odds = sum(1 for x in test_selected if x % 2 == 1)
                evens = len(test_selected) - odds
                if odds > 4 or evens > 4:
                    continue

                # 最後の数字: 合計値チェック（改善⑥: ±1σ）
                # 後半のトライアルほど許容幅を広げ、制約が厳しすぎて解が全滅
                # （予想が空）になるのを防ぐ。品質スコアが中央寄りを優遇するため、
                # 余裕がある場合は自然とタイトな合計値の解が選ばれる。
                if len(test_selected) == 5:
                    s = sum(test_selected)
                    relax = 1.0 + trial * 0.15  # trial0: ±1σ 〜 trial19: ±約3.85σ
                    lo = avg_sum - std_sum * relax
                    hi = avg_sum + std_sum * relax
                    if not (lo <= s <= hi):
                        continue

                selected.append(n)

                # 理由生成用にスコアを記録
                factor_scores = {
                    "freq": scores["freq"] * weights["freq"],
                    "recent": scores["recent"] * weights["recent"],
                    "drought": scores["drought"] * weights["drought"],
                    "pull": scores["pull"] * weights["pull"],
                    "pair": scores["pair"] * weights["pair"],
                    "consec": scores["consec"] * weights["consec"],
                    "cycle": scores["cycle"] * weights["cycle"],
                    "rf": scores["rf"] * weights["rf"],
                    "lstm": scores["lstm"] * weights["lstm"],
                }
                sel_reasons[str(n)] = {
                    "total_score": total_score,
                    "factor_scores": factor_scores,
                    "raw_scores": {k: v for k, v in scores.items() if k != "pair_raw"},
                }
                found = True
                break

            if not found:
                break

        if len(selected) != 5:
            continue

        # この候補セットの品質を評価
        trial_sum = sum(selected)
        sum_deviation = abs(trial_sum - avg_sum) / std_sum
        zones = Counter(get_zone(x) for x in selected)
        zone_balance = 1.0 / (1.0 + max(zones.values()) - min(zones.values()))
        odds = sum(1 for x in selected if x % 2 == 1)
        odd_balance = 1.0 - abs(odds - 3) / 3.0
        total_quality = sum(sel_reasons[str(n)]["total_score"] for n in selected)
        avg_quality = total_quality / 5.0
        quality = avg_quality * DIVERSITY_QUALITY_W + zone_balance * 0.2 + odd_balance * 0.2 + (1.0 - min(sum_deviation, 2.0) / 2.0) * 0.2 + random.random() * DIVERSITY_JITTER

        if quality > best_score:
            best_score = quality
            best_result = (selected[:], dict(sel_reasons))

    if best_result is None:
        # 制約(合計値・帯・奇偶)を全て満たす組み合わせが見つからなかった場合でも
        # 予想を空にせず、ベーススコア上位5個を採用する（「データ不足」表示の防止）。
        top5 = sorted(range(1, 32), key=lambda n: base_scores.get(n, 0), reverse=True)[:5]
        fb_reasons = {}
        for n in top5:
            raw = {
                "freq": freq_scores.get(n, 0),
                "recent": recent_scores.get(n, 0),
                "drought": drought_scores.get(n, 0),
                "pull": pull_scores.get(n, 0),
                "pair": 0.0,
                "consec": consec_base_scores.get(n, 0),
                "cycle": cycle_scores.get(n, 0),
                "rf": rf_scores.get(n, 0),
                "lstm": lstm_scores.get(n, 0),
                "random": 0.0,
            }
            fb_reasons[str(n)] = {
                "total_score": base_scores.get(n, 0),
                "factor_scores": {k: raw[k] * weights[k] for k in weights if k != "random"},
                "raw_scores": raw,
            }
        best_result = (top5, fb_reasons)
        print(f"  [fallback] {mode_key}/{period_label}: 制約解なし→ベーススコア上位5個で補完")

    selected, sel_reasons = best_result
    selected.sort()

    # --- 選出理由の文章生成 ---
    reasons = {}
    for n in selected:
        info = sel_reasons[str(n)]
        factor_scores = info["factor_scores"]
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        top_factor = sorted_factors[0][0]
        second_factor = sorted_factors[1][0] if len(sorted_factors) > 1 else None

        freq_count = freq_data["counts"].get(str(n), 0)
        freq_pct = freq_data["percentages"].get(str(n), 0)
        r100_pct = freq_data["recent_100"].get(str(n), 0)
        r300_pct = freq_data["recent_300"].get(str(n), 0)
        drought_val = freq_data["drought"].get(str(n), 0)
        avg_interval = freq_data["avg_intervals"].get(str(n), 7)

        reason_parts = []

        # メイン理由
        if top_factor == "freq":
            reason_parts.append(f"全{total_draws}回中{freq_count}回出現（{freq_pct}%）と高い出現頻度を記録")
        elif top_factor == "recent":
            reason_parts.append(f"直近100回で{r100_pct}%と最近の出現率が高い（全期間{freq_pct}%）")
        elif top_factor == "drought":
            ratio = round(drought_val / avg_interval, 1) if avg_interval > 0 else 0
            reason_parts.append(f"平均{avg_interval:.0f}回間隔に対し{drought_val}回未出現（{ratio}倍の遅延）で出現期待が高い")
        elif top_factor == "pull":
            recent_count = sum(1 for d in draws[-5:] if n in d["numbers"])
            reason_parts.append(f"直近5回中{recent_count}回出現しており、連続出現の勢いあり")
        elif top_factor == "pair":
            reason_parts.append("選出済みの他の数字との同時出現回数が多く、相性が良い")
        elif top_factor == "consec":
            neighbors = [s for s in selected if abs(n - s) == 1 and s != n]
            if neighbors:
                reason_parts.append(f"{neighbors[0]}との連番ペアで出現しやすい傾向")
            else:
                reason_parts.append("連番を含む抽選で出やすい傾向がある")
        elif top_factor == "cycle":
            cd = cycle_data.get(str(n), {})
            dc = cd.get("dominant_cycle", "?")
            reason_parts.append(f"約{dc}回周期で出現するパターンを検出、次の出現タイミングに該当")
        elif top_factor == "rf":
            rf_prob = rf_scores_raw.get(n, 0.5)
            reason_parts.append(f"ランダムフォレストが出現確率{rf_prob:.1%}と高く予測")
        elif top_factor == "lstm":
            lstm_prob = lstm_scores_raw.get(n, 0.5)
            reason_parts.append(f"LSTMが時系列パターンから出現確率{lstm_prob:.1%}と予測")

        # サブ理由（2番目に効いた要素）
        if second_factor and second_factor != top_factor:
            if second_factor == "recent" and r100_pct > float(freq_pct):
                reason_parts.append(f"直近100回の出現率{r100_pct}%で上昇傾向")
            elif second_factor == "drought" and drought_val >= 8:
                ratio = round(drought_val / avg_interval, 1) if avg_interval > 0 else 0
                reason_parts.append(f"平均間隔の{ratio}倍となる{drought_val}回未出現")
            elif second_factor == "pull" and n in last_numbers:
                reason_parts.append("前回の抽選でも出現")
            elif second_factor == "freq" and float(freq_pct) > 14.0:
                reason_parts.append(f"全期間出現率{freq_pct}%と安定して高い")
            elif second_factor == "pair":
                reason_parts.append("他の選出数字との相性も良好")
            elif second_factor == "consec":
                neighbors = [s for s in selected if abs(n - s) == 1 and s != n]
                if neighbors:
                    reason_parts.append(f"{neighbors[0]}と連番")
            elif second_factor == "cycle":
                cd = cycle_data.get(str(n), {})
                reason_parts.append(f"約{cd.get('dominant_cycle', '?')}回周期のタイミング")
            elif second_factor == "rf":
                reason_parts.append(f"RF予測でも高評価")
            elif second_factor == "lstm":
                reason_parts.append(f"LSTM予測でも高評価")

        zone_name = {"low": "低帯(1-10)", "mid": "中帯(11-21)", "high": "高帯(22-31)"}[get_zone(n)]
        reason_parts.append(zone_name)

        reason_text = "。".join(reason_parts) + "。"

        reasons[str(n)] = {
            "score": round(info["total_score"], 4),
            "top_factor": top_factor,
            "reason_text": reason_text,
            "details": {k: round(v, 4) for k, v in info["raw_scores"].items()},
            "monte_carlo_pct": mc_confidence.get(n, 0.0),
        }

    # --- ボーナス数字選出 ---
    bonus_candidates = []
    for n in range(1, 32):
        if n in selected:
            continue
        pair_score = 0.0
        for s in selected:
            key = f"{min(n,s)}-{max(n,s)}"
            all_c = pair_counts_all.get(key, 0)
            rec_c = pair_counts_recent.get(key, 0) * (total_draws / min(200, total_draws))
            pair_score += all_c * 0.6 + rec_c * 0.4
        pair_score /= len(selected)

        total = (
            weights["freq"] * freq_scores.get(n, 0)
            + weights["recent"] * recent_scores.get(n, 0)
            + weights["drought"] * drought_scores.get(n, 0)
            + weights["pull"] * pull_scores.get(n, 0)
            + weights["consec"] * consec_base_scores.get(n, 0)
            + weights["cycle"] * cycle_scores.get(n, 0)
            + weights["rf"] * rf_scores.get(n, 0)
            + weights["lstm"] * lstm_scores.get(n, 0)
            + weights["random"] * random.random()
        )
        bonus_candidates.append((n, total))
    bonus_candidates.sort(key=lambda x: x[1], reverse=True)
    bonus_number = bonus_candidates[0][0] if bonus_candidates else None

    # ボーナス数字の理由文
    bonus_reason = ""
    if bonus_number:
        bn = bonus_number
        b_freq_pct = freq_data["percentages"].get(str(bn), 0)
        b_drought = freq_data["drought"].get(str(bn), 0)
        b_avg_int = freq_data["avg_intervals"].get(str(bn), 7)
        b_r100 = freq_data["recent_100"].get(str(bn), 0)
        b_parts = ["本数字5個に次ぐ総合スコアで選出"]
        if float(b_r100) > float(b_freq_pct):
            b_parts.append(f"直近100回の出現率{b_r100}%で上昇傾向")
        elif float(b_freq_pct) > 14:
            b_parts.append(f"出現率{b_freq_pct}%と高頻度")
        if b_drought >= 8:
            ratio = round(b_drought / b_avg_int, 1) if b_avg_int > 0 else 0
            b_parts.append(f"平均間隔の{ratio}倍（{b_drought}回）未出現")
        if bn in last_numbers:
            b_parts.append("前回も出現")
        bonus_reason = "。".join(b_parts) + "。"

    # メトリクス
    odds = sum(1 for x in selected if x % 2 == 1)
    evens = 5 - odds
    zones = Counter(get_zone(x) for x in selected)
    total_sum = sum(selected)

    return {
        "numbers": selected,
        "bonus": bonus_number,
        "bonus_reason": bonus_reason,
        "reasons": reasons,
        "metrics": {
            "odd_even": f"{odds}:{evens}",
            "zones": f"{zones.get('low',0)}-{zones.get('mid',0)}-{zones.get('high',0)}",
            "sum": total_sum,
            "avg_sum": round(avg_sum, 1),
            "sum_std": round(std_sum, 1),
            "sum_range": f"{sum_range[0]:.0f}〜{sum_range[1]:.0f}",
        },
        "monte_carlo": {str(k): v for k, v in mc_confidence.items()},
    }


def run_predictions(freq_data, pull_data, zone_data, pair_data, consec_data, cycle_data, rf_scores, lstm_scores, draws, period_label="all"):
    predictions = {}
    for mode_key, weights in MODES.items():
        random.seed(f"{date.today().isoformat()}_{mode_key}_{period_label}")
        predictions[mode_key] = generate_prediction(
            freq_data, pull_data, zone_data, pair_data, consec_data, cycle_data, rf_scores, lstm_scores, draws, weights,
            mode_key=mode_key, period_label=period_label,
        )
        predictions[mode_key]["mode_name"] = MODE_NAMES[mode_key]
    return predictions


def analyze_period(draws, period_label="all"):
    """指定された抽選データに対して全分析 + 予想を実行"""
    print(f"  Frequency/Pull/Zone/Pair/Consecutive...")
    freq = analyze_frequency(draws)
    pull = analyze_pull(draws)
    zone = analyze_zone(draws)
    pairs = analyze_pairs(draws)
    consec = analyze_consecutive(draws)

    print(f"  Cycle analysis...")
    cycle = analyze_cycle(draws)

    print(f"  Random Forest...")
    rf_scores = predict_rf(draws)

    print(f"  LSTM...")
    lstm_scores = predict_lstm(draws)

    print(f"  Generating predictions...")
    predictions = run_predictions(freq, pull, zone, pairs, consec, cycle, rf_scores, lstm_scores, draws, period_label)

    sums = [sum(d["numbers"]) for d in draws]
    odds_counts = [sum(1 for n in d["numbers"] if n % 2 == 1) for d in draws]

    return {
        "frequency": freq,
        "pull": pull,
        "zone": zone,
        "consecutive": consec,
        "pairs": {
            "top_pairs": pairs["top_pairs"],
            "affinity": pairs["affinity"],
        },
        "cycle": cycle,
        "rf_scores": {str(k): round(v, 4) for k, v in rf_scores.items()},
        "lstm_scores": {str(k): round(v, 4) for k, v in lstm_scores.items()},
        "predictions": predictions,
        "summary_stats": {
            "total_draws": len(draws),
            "avg_sum": round(sum(sums) / len(sums), 1),
            "sum_std": round(math.sqrt(sum((s - sum(sums) / len(sums)) ** 2 for s in sums) / len(sums)), 1),
            "avg_odd_count": round(sum(odds_counts) / len(odds_counts), 1),
            "date_range": [draws[0]["date"], draws[-1]["date"]],
        },
        "recent_draws": [
            {
                "round": d["round"],
                "date": d["date"],
                "numbers": d["numbers"],
                "bonus": d["bonus"],
                "sum": sum(d["numbers"]),
                "odd_even": f"{sum(1 for n in d['numbers'] if n % 2 == 1)}:{sum(1 for n in d['numbers'] if n % 2 == 0)}",
                "zones": f"{sum(1 for n in d['numbers'] if n <= 10)}-{sum(1 for n in d['numbers'] if 11 <= n <= 21)}-{sum(1 for n in d['numbers'] if n >= 22)}",
            }
            for d in draws[-20:]
        ],
    }


# ============================================================
# メイン
# ============================================================
PERIOD_SIZES = [100, 200, 300, 400]  # 直近N回


def compute_periods(draws):
    """与えられた抽選データから全期間＋各直近N回の分析・予想を計算し、(periods, period_labels)を返す。"""
    periods = {"all": analyze_period(draws, "all")}
    for size in PERIOD_SIZES:
        if len(draws) < size:
            continue
        periods[str(size)] = analyze_period(draws[-size:], str(size))

    period_labels = []
    for size in PERIOD_SIZES:
        if str(size) in periods:
            pd = draws[-size:]
            period_labels.append({
                "key": str(size),
                "label": f"直近{size}回",
                "range": f"第{pd[0]['round']}回〜第{pd[-1]['round']}回",
                "draws": size,
            })
    period_labels.append({
        "key": "all",
        "label": "全期間",
        "range": f"第{draws[0]['round']}回〜第{draws[-1]['round']}回",
        "draws": len(draws),
    })
    return periods, period_labels


def backfill_missing_archive(archive, all_draws, last_updated):
    """実際の結果はあるのに予想記録が欠けている回を、その時点(直前回まで)のデータで
    再構築して補完する。取得元の掲載漏れ等で予想が生成されなかった回の穴埋め用。"""
    if not archive:
        return
    archived = {e["predicted_round"] for e in archive}
    draw_rounds = {d["round"] for d in all_draws}
    latest = all_draws[-1]["round"]
    missing = sorted(
        r for r in range(min(archived), latest + 1)
        if r in draw_rounds and r not in archived
    )
    for r in missing:
        hist_draws = [d for d in all_draws if d["round"] < r]
        if len(hist_draws) < max(PERIOD_SIZES):
            continue
        print(f"Backfilling missing archive entry for round {r} (data up to {hist_draws[-1]['round']})")
        hp, hpl = compute_periods(hist_draws)
        archive.append(build_archive_entry(r, hist_draws[-1]["round"], hp, hpl, last_updated))


def repair_empty_predictions(archive, all_draws, last_updated):
    """過去のアーカイブ記録のうち、制約解なしで予想が空(numbers=[])になっている
    モードだけを、その時点(直前回まで)のデータで再計算して埋める。既存の非空予想には
    一切触れず、空の枠のみ差し替えることで履歴の整合性を保つ。"""
    if not archive:
        return
    for entry in archive:
        pr = entry["predicted_round"]
        empties = [
            (pk, mk)
            for pk, pdata in entry["predictions_by_period"].items()
            for mk, pred in pdata["modes"].items()
            if not pred.get("numbers")
        ]
        if not empties:
            continue
        hist_draws = [d for d in all_draws if d["round"] < pr]
        if len(hist_draws) < max(PERIOD_SIZES):
            continue
        print(f"Repairing {len(empties)} empty prediction(s) in round {pr}: {empties}")
        hp, hpl = compute_periods(hist_draws)
        fresh = build_archive_entry(pr, hist_draws[-1]["round"], hp, hpl, last_updated)
        changed = False
        for pk, mk in empties:
            new_mode = fresh["predictions_by_period"].get(pk, {}).get("modes", {}).get(mk)
            if new_mode and new_mode.get("numbers"):
                entry["predictions_by_period"][pk]["modes"][mk] = new_mode
                changed = True
        # 新しく埋めた予想の答え合わせを行うため、この回だけ再検証させる
        if changed:
            entry["verified"] = False


def main():
    print("=== MiniLoto Analyzer v2 (マルチ期間対応) ===")
    data = load_data()
    all_draws = data["draws"]
    print(f"Loaded {len(all_draws)} draws")

    # 全期間＋各直近N回の分析・予想
    print(f"\n--- 全期間 ({len(all_draws)}回) ＋ 各直近N回の分析 ---")
    periods, period_labels = compute_periods(all_draws)

    # ============================================================
    # アーカイブ: 予想を保存 + 答え合わせ
    # ============================================================
    archive_path = OUTPUT_PATH.parent / "archive.json"
    archive = load_archive(archive_path)
    latest_round = all_draws[-1]["round"]
    next_round = latest_round + 1

    # 欠けている過去の予想記録を補完（取得元の掲載漏れ等で予想が生成されなかった回）
    backfill_missing_archive(archive, all_draws, data["last_updated"])

    # 制約解なしで空になっていた過去の予想を補完（該当枠のみ再計算）
    repair_empty_predictions(archive, all_draws, data["last_updated"])

    # 答え合わせ: アーカイブ済みの予想に実際の結果を突き合わせ
    verify_archive(archive, all_draws)

    # 今回の予想をアーカイブに追加（まだ保存されていない場合のみ）
    if not any(e["predicted_round"] == next_round for e in archive):
        entry = build_archive_entry(next_round, latest_round, periods, period_labels, data["last_updated"])
        archive.append(entry)
        print(f"\nArchived predictions for round {next_round}")
    else:
        print(f"\nRound {next_round} already archived, skipping")

    # アーカイブ保存（回号順に整列）
    archive.sort(key=lambda e: e["predicted_round"])
    save_archive(archive_path, archive)

    # モード別累計成績
    mode_stats = calc_mode_stats(archive)

    output = {
        "last_updated": data["last_updated"],
        "latest_round": latest_round,
        "period_labels": period_labels,
        "periods": periods,
        "archive": archive,
        "mode_stats": mode_stats,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nAnalysis saved to {OUTPUT_PATH}")

    # 結果表示
    for period_key, result in periods.items():
        label = f"直近{period_key}回" if period_key != "all" else "全期間"
        print(f"\n=== {label} ===")
        for mode_key, pred in result["predictions"].items():
            print(f"  {pred['mode_name']}: {pred['numbers']} + bonus:{pred['bonus']}")


# ============================================================
# アーカイブ関連関数
# ============================================================
ARCHIVE_PATH = Path(__file__).parent.parent / "docs" / "data" / "archive.json"


def load_archive(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_archive(path, archive):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(archive, f, ensure_ascii=False, indent=2)


def build_archive_entry(predicted_round, data_round, periods, period_labels, last_updated):
    """全期間×全モードの予想をアーカイブエントリとして構築"""
    entry = {
        "predicted_round": predicted_round,
        "data_up_to_round": data_round,
        "generated_at": last_updated,
        "actual": None,  # 結果が出たら埋まる
        "verified": False,
        "predictions_by_period": {},
    }

    for pinfo in period_labels:
        pkey = pinfo["key"]
        pdata = periods.get(pkey)
        if not pdata:
            continue

        period_preds = {}
        for mode_key, pred in pdata["predictions"].items():
            period_preds[mode_key] = {
                "mode_name": pred["mode_name"],
                "numbers": pred["numbers"],
                "bonus": pred.get("bonus"),
                "bonus_reason": pred.get("bonus_reason", ""),
                "reasons": pred.get("reasons", {}),
                "metrics": pred.get("metrics", {}),
                "match_count": None,  # 答え合わせ後に埋まる
                "matched_numbers": None,
                "bonus_matched": None,
            }
        entry["predictions_by_period"][pkey] = {
            "label": pinfo["label"],
            "range": pinfo["range"],
            "draws": pinfo["draws"],
            "modes": period_preds,
        }

    return entry


def verify_archive(archive, draws):
    """アーカイブの予想と実際の結果を突き合わせ"""
    draw_map = {d["round"]: d for d in draws}

    for entry in archive:
        if entry["verified"]:
            continue

        pred_round = entry["predicted_round"]
        if pred_round not in draw_map:
            continue  # まだ抽選されていない

        actual = draw_map[pred_round]
        actual_set = set(actual["numbers"])
        actual_bonus = actual["bonus"]

        entry["actual"] = {
            "numbers": actual["numbers"],
            "bonus": actual_bonus,
            "date": actual["date"],
        }
        entry["verified"] = True

        # 各期間×各モードの答え合わせ
        for pkey, pdata in entry["predictions_by_period"].items():
            for mode_key, pred in pdata["modes"].items():
                pred_set = set(pred["numbers"])
                matched = sorted(list(pred_set & actual_set))
                pred["match_count"] = len(matched)
                pred["matched_numbers"] = matched
                pred["bonus_matched"] = pred.get("bonus") == actual_bonus

        print(f"  Verified round {pred_round}: actual={actual['numbers']} bonus={actual_bonus}")


def calc_mode_stats(archive):
    """モード別の累計成績を集計（ミニロトは1〜4等の4段階）"""
    verified = [e for e in archive if e["verified"]]
    if not verified:
        return {}

    stats = {}
    for entry in verified:
        for pkey, pdata in entry["predictions_by_period"].items():
            if pkey not in stats:
                stats[pkey] = {"label": pdata["label"], "modes": {}}
            for mode_key, pred in pdata["modes"].items():
                if mode_key not in stats[pkey]["modes"]:
                    stats[pkey]["modes"][mode_key] = {
                        "mode_name": pred["mode_name"],
                        "total_rounds": 0,
                        "match_distribution": {str(i): 0 for i in range(6)},
                        "total_matched": 0,
                        "bonus_matched": 0,
                        "best_match": 0,
                        "prize_counts": {"1st": 0, "2nd": 0, "3rd": 0, "4th": 0},
                    }
                s = stats[pkey]["modes"][mode_key]
                mc = pred["match_count"] if pred["match_count"] is not None else 0
                s["total_rounds"] += 1
                s["match_distribution"][str(mc)] = s["match_distribution"].get(str(mc), 0) + 1
                s["total_matched"] += mc
                if pred.get("bonus_matched"):
                    s["bonus_matched"] += 1
                if mc > s["best_match"]:
                    s["best_match"] = mc
                # 等級判定（1等:5個一致 / 2等:4個一致+ボーナス / 3等:4個一致 / 4等:3個一致）
                bonus_hit = pred.get("bonus_matched", False)
                if mc == 5:
                    s["prize_counts"]["1st"] += 1
                elif mc == 4 and bonus_hit:
                    s["prize_counts"]["2nd"] += 1
                elif mc == 4:
                    s["prize_counts"]["3rd"] += 1
                elif mc == 3:
                    s["prize_counts"]["4th"] += 1

    # モンテカルロ・シミュレーションによるランダム基準（AIモードとの比較用）
    for pkey, pdata in stats.items():
        any_mode = next(iter(pdata["modes"].values()), None)
        if any_mode:
            pdata["random_baseline"] = simulate_random_baseline(
                any_mode["total_rounds"], n_sim=200000, seed_str=f"baseline_{pkey}_{any_mode['total_rounds']}"
            )

    return stats


if __name__ == "__main__":
    main()
