"""WorldQuant '101 Formulaic Alphas' (Kakushadze, 2015), translated into AlphaMine's DSL.

Use as a large, ready-made seed bank — you supply nothing.

Translation notes (read once):
  - adv{d}  (average daily volume) -> ts_mean(volume, d)
  - vwap is taken from the data panel (synthetic uses (h+l+c)/3; real data should use a
    true VWAP if you have it).
  - IndNeutralize(x, IndClass.*) -> indneutralize(x), which we APPROXIMATE as a daily
    cross-sectional demean (we don't ship sector labels). Alphas relying on this are marked
    approx=True below; swap in real sector neutralization for production use.
  - Non-integer window constants from the paper (e.g. 8.93345) are rounded to ints, since
    rolling windows must be integers. This is standard practice and barely changes behavior.
  - Ternaries (a ? b : c) -> iff(cond, b, c); comparisons -> lt/gt/le/ge.
  - Alpha #56 uses market cap, which we don't have -> omitted (listed in SKIPPED).

Each item: (id, expr, approx)  where approx=True means IndNeutralize was approximated.
"""
from __future__ import annotations

from typing import List, Tuple

from .alpha import Alpha, AlphaError, validate

# id -> reason it is not included
SKIPPED = {56: "requires market cap (`cap`), not available in OHLCV data"}

# (id, expression, approx_indneutralize)
ALPHA101: List[Tuple[int, str, bool]] = [
    (1, "rank(ts_argmax(signedpower(iff(lt(returns, 0), ts_std(returns, 20), close), 2.0), 5)) - 0.5", False),
    (2, "-1 * corr(rank(delta(log(volume), 2)), rank((close - open) / open), 6)", False),
    (3, "-1 * corr(rank(open), rank(volume), 10)", False),
    (4, "-1 * ts_rank(rank(low), 9)", False),
    (5, "rank(open - (ts_sum(vwap, 10) / 10)) * (-1 * abs_(rank(close - vwap)))", False),
    (6, "-1 * corr(open, volume, 10)", False),
    (7, "iff(lt(ts_mean(volume, 20), volume), (-1 * ts_rank(abs_(delta(close, 7)), 60)) * sign(delta(close, 7)), -1)", False),
    (8, "-1 * rank((ts_sum(open, 5) * ts_sum(returns, 5)) - delay(ts_sum(open, 5) * ts_sum(returns, 5), 10))", False),
    (9, "iff(lt(0, ts_min(delta(close, 1), 5)), delta(close, 1), iff(lt(ts_max(delta(close, 1), 5), 0), delta(close, 1), -1 * delta(close, 1)))", False),
    (10, "rank(iff(lt(0, ts_min(delta(close, 1), 4)), delta(close, 1), iff(lt(ts_max(delta(close, 1), 4), 0), delta(close, 1), -1 * delta(close, 1))))", False),
    (11, "(rank(ts_max(vwap - close, 3)) + rank(ts_min(vwap - close, 3))) * rank(delta(volume, 3))", False),
    (12, "sign(delta(volume, 1)) * (-1 * delta(close, 1))", False),
    (13, "-1 * rank(cov(rank(close), rank(volume), 5))", False),
    (14, "(-1 * rank(delta(returns, 3))) * corr(open, volume, 10)", False),
    (15, "-1 * ts_sum(rank(corr(rank(high), rank(volume), 3)), 3)", False),
    (16, "-1 * rank(cov(rank(high), rank(volume), 5))", False),
    (17, "((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank(volume / ts_mean(volume, 20), 5))", False),
    (18, "-1 * rank((ts_std(abs_(close - open), 5) + (close - open)) + corr(close, open, 10))", False),
    (19, "(-1 * sign((close - delay(close, 7)) + delta(close, 7))) * (1 + rank(1 + ts_sum(returns, 250)))", False),
    (20, "((-1 * rank(open - delay(high, 1))) * rank(open - delay(close, 1))) * rank(open - delay(low, 1))", False),
    (21, "iff(lt((ts_sum(close, 8) / 8) + ts_std(close, 8), ts_sum(close, 2) / 2), -1, iff(lt(ts_sum(close, 2) / 2, (ts_sum(close, 8) / 8) - ts_std(close, 8)), 1, iff(ge(volume / ts_mean(volume, 20), 1), 1, -1)))", False),
    (22, "-1 * (delta(corr(high, volume, 5), 5) * rank(ts_std(close, 20)))", False),
    (23, "iff(lt(ts_sum(high, 20) / 20, high), -1 * delta(high, 2), 0)", False),
    (24, "iff(le(delta(ts_sum(close, 100) / 100, 100) / delay(close, 100), 0.05), -1 * (close - ts_min(close, 100)), -1 * delta(close, 3))", False),
    (25, "rank((((-1 * returns) * ts_mean(volume, 20)) * vwap) * (high - close))", False),
    (26, "-1 * ts_max(corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)", False),
    (27, "iff(lt(0.5, rank(ts_sum(corr(rank(volume), rank(vwap), 6), 2) / 2.0)), -1, 1)", False),
    (28, "scale((corr(ts_mean(volume, 20), low, 5) + ((high + low) / 2)) - close)", False),
    (29, "ts_min(ts_product(rank(rank(scale(log(ts_sum(ts_min(rank(rank(-1 * rank(delta(close - 1, 5)))), 2), 1))))), 1), 5) + ts_rank(delay(-1 * returns, 6), 5)", False),
    (30, "((1.0 - rank((sign(close - delay(close, 1)) + sign(delay(close, 1) - delay(close, 2))) + sign(delay(close, 2) - delay(close, 3)))) * ts_sum(volume, 5)) / ts_sum(volume, 20)", False),
    (31, "(rank(rank(rank(decay_linear(-1 * rank(rank(delta(close, 10))), 10)))) + rank(-1 * delta(close, 3))) + sign(scale(corr(ts_mean(volume, 20), low, 12)))", False),
    (32, "scale((ts_sum(close, 7) / 7) - close) + (20 * scale(corr(vwap, delay(close, 5), 230)))", False),
    (33, "rank(-1 * ((1 - (open / close)) ** 1))", False),
    (34, "rank((1 - rank(ts_std(returns, 2) / ts_std(returns, 5))) + (1 - rank(delta(close, 1))))", False),
    (35, "(ts_rank(volume, 32) * (1 - ts_rank((close + high) - low, 16))) * (1 - ts_rank(returns, 32))", False),
    (36, "((((2.21 * rank(corr(close - open, delay(volume, 1), 15))) + (0.7 * rank(open - close))) + (0.73 * rank(ts_rank(delay(-1 * returns, 6), 5)))) + rank(abs_(corr(vwap, ts_mean(volume, 20), 6)))) + (0.6 * rank(((ts_sum(close, 200) / 200) - open) * (close - open)))", False),
    (37, "rank(corr(delay(open - close, 1), close, 200)) + rank(open - close)", False),
    (38, "(-1 * rank(ts_rank(close, 10))) * rank(close / open)", False),
    (39, "(-1 * rank(delta(close, 7) * (1 - rank(decay_linear(volume / ts_mean(volume, 20), 9))))) * (1 + rank(ts_sum(returns, 250)))", False),
    (40, "(-1 * rank(ts_std(high, 10))) * corr(high, volume, 10)", False),
    (41, "((high * low) ** 0.5) - vwap", False),
    (42, "rank(vwap - close) / rank(vwap + close)", False),
    (43, "ts_rank(volume / ts_mean(volume, 20), 20) * ts_rank(-1 * delta(close, 7), 8)", False),
    (44, "-1 * corr(high, rank(volume), 5)", False),
    (45, "-1 * ((rank(ts_sum(delay(close, 5), 20) / 20) * corr(close, volume, 2)) * rank(corr(ts_sum(close, 5), ts_sum(close, 20), 2)))", False),
    (46, "iff(lt(0.25, ((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)), -1, iff(lt(((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10), 0), 1, -1 * (close - delay(close, 1))))", False),
    (47, "(((rank(1 / close) * volume) / ts_mean(volume, 20)) * ((high * rank(high - close)) / (ts_sum(high, 5) / 5))) - rank(vwap - delay(vwap, 5))", False),
    (48, "indneutralize((corr(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close) / ts_sum((delta(close, 1) / delay(close, 1)) ** 2, 250)", True),
    (49, "iff(lt(((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10), -0.1), 1, -1 * (close - delay(close, 1)))", False),
    (50, "-1 * ts_max(rank(corr(rank(volume), rank(vwap), 5)), 5)", False),
    (51, "iff(lt(((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10), -0.05), 1, -1 * (close - delay(close, 1)))", False),
    (52, "(((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220)) * ts_rank(volume, 5)", False),
    (53, "-1 * delta(((close - low) - (high - close)) / (close - low), 9)", False),
    (54, "(-1 * ((low - close) * (open ** 5))) / ((low - high) * (close ** 5))", False),
    (55, "-1 * corr(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), rank(volume), 6)", False),
    (57, "0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)))", False),
    (58, "-1 * ts_rank(decay_linear(corr(indneutralize(vwap), volume, 4), 8), 6)", True),
    (59, "-1 * ts_rank(decay_linear(corr(indneutralize(vwap), volume, 4), 16), 8)", True),
    (60, "0 - (1 * ((2 * scale(rank((((close - low) - (high - close)) / (high - low)) * volume))) - scale(rank(ts_argmax(close, 10)))))", False),
    (61, "lt(rank(vwap - ts_min(vwap, 16)), rank(corr(vwap, ts_mean(volume, 180), 18)))", False),
    (62, "lt(rank(corr(vwap, ts_sum(ts_mean(volume, 20), 22), 10)), rank(lt(rank(open) + rank(open), rank((high + low) / 2) + rank(high)))) * -1", False),
    (63, "(rank(decay_linear(delta(indneutralize(close), 2), 8)) - rank(decay_linear(corr((vwap * 0.318108) + (open * (1 - 0.318108)), ts_sum(ts_mean(volume, 180), 37), 14), 12))) * -1", True),
    (64, "lt(rank(corr(ts_sum((open * 0.178404) + (low * (1 - 0.178404)), 13), ts_sum(ts_mean(volume, 120), 13), 17)), rank(delta((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404)), 4))) * -1", False),
    (65, "lt(rank(corr((open * 0.00817205) + (vwap * (1 - 0.00817205)), ts_sum(ts_mean(volume, 60), 9), 6)), rank(open - ts_min(open, 14))) * -1", False),
    (66, "(rank(decay_linear(delta(vwap, 4), 7)) + ts_rank(decay_linear((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2)), 11), 7)) * -1", False),
    (67, "(rank(high - ts_min(high, 2)) ** rank(corr(indneutralize(vwap), indneutralize(ts_mean(volume, 20)), 6))) * -1", True),
    (68, "lt(ts_rank(corr(rank(high), rank(ts_mean(volume, 15)), 9), 14), rank(delta((close * 0.518371) + (low * (1 - 0.518371)), 1))) * -1", False),
    (69, "(rank(ts_max(delta(indneutralize(vwap), 3), 5)) ** ts_rank(corr((close * 0.490655) + (vwap * (1 - 0.490655)), ts_mean(volume, 20), 5), 9)) * -1", True),
    (70, "(rank(delta(vwap, 1)) ** ts_rank(corr(indneutralize(close), ts_mean(volume, 50), 18), 18)) * -1", True),
    (71, "max2(ts_rank(decay_linear(corr(ts_rank(close, 3), ts_rank(ts_mean(volume, 180), 12), 18), 4), 16), ts_rank(decay_linear(rank((low + open) - (vwap + vwap)) ** 2, 16), 4))", False),
    (72, "rank(decay_linear(corr((high + low) / 2, ts_mean(volume, 40), 9), 10)) / rank(decay_linear(corr(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))", False),
    (73, "max2(rank(decay_linear(delta(vwap, 5), 3)), ts_rank(decay_linear((delta((open * 0.147155) + (low * (1 - 0.147155)), 2) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1, 3), 17)) * -1", False),
    (74, "lt(rank(corr(close, ts_sum(ts_mean(volume, 30), 37), 15)), rank(corr(rank((high * 0.0261661) + (vwap * (1 - 0.0261661))), rank(volume), 11))) * -1", False),
    (75, "lt(rank(corr(vwap, volume, 4)), rank(corr(rank(low), rank(ts_mean(volume, 50)), 12)))", False),
    (76, "max2(rank(decay_linear(delta(vwap, 1), 12)), ts_rank(decay_linear(ts_rank(corr(indneutralize(low), ts_mean(volume, 81), 8), 20), 17), 19)) * -1", True),
    (77, "min2(rank(decay_linear((((high + low) / 2) + high) - (vwap + high), 20)), rank(decay_linear(corr((high + low) / 2, ts_mean(volume, 40), 3), 6)))", False),
    (78, "rank(corr(ts_sum((low * 0.352233) + (vwap * (1 - 0.352233)), 20), ts_sum(ts_mean(volume, 40), 20), 7)) ** rank(corr(rank(vwap), rank(volume), 6))", False),
    (79, "lt(rank(delta(indneutralize((close * 0.60733) + (open * (1 - 0.60733))), 1)), rank(corr(ts_rank(vwap, 4), ts_rank(ts_mean(volume, 150), 9), 15)))", True),
    (80, "(rank(sign(delta(indneutralize((open * 0.868128) + (high * (1 - 0.868128))), 4))) ** ts_rank(corr(high, ts_mean(volume, 10), 5), 6)) * -1", True),
    (81, "lt(rank(log(ts_product(rank(rank(corr(vwap, ts_sum(ts_mean(volume, 10), 50), 8)) ** 4), 15))), rank(corr(rank(vwap), rank(volume), 5))) * -1", False),
    (82, "min2(rank(decay_linear(delta(open, 1), 15)), ts_rank(decay_linear(corr(indneutralize(volume), open, 17), 7), 13)) * -1", True),
    (83, "(rank(delay((high - low) / (ts_sum(close, 5) / 5), 2)) * rank(rank(volume))) / (((high - low) / (ts_sum(close, 5) / 5)) / (vwap - close))", False),
    (84, "signedpower(ts_rank(vwap - ts_max(vwap, 15), 21), delta(close, 5))", False),
    (85, "rank(corr((high * 0.876703) + (close * (1 - 0.876703)), ts_mean(volume, 30), 10)) ** rank(corr(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7))", False),
    (86, "lt(ts_rank(corr(close, ts_sum(ts_mean(volume, 20), 15), 6), 20), rank((open + close) - (vwap + open))) * -1", False),
    (87, "max2(rank(decay_linear(delta((close * 0.369701) + (vwap * (1 - 0.369701)), 2), 3)), ts_rank(decay_linear(abs_(corr(indneutralize(ts_mean(volume, 81)), close, 13)), 5), 14)) * -1", True),
    (88, "min2(rank(decay_linear((rank(open) + rank(low)) - (rank(high) + rank(close)), 8)), ts_rank(decay_linear(corr(ts_rank(close, 8), ts_rank(ts_mean(volume, 60), 21), 8), 7), 3))", False),
    (89, "ts_rank(decay_linear(corr(low, ts_mean(volume, 10), 7), 6), 4) - ts_rank(decay_linear(delta(indneutralize(vwap), 3), 10), 15)", True),
    (90, "(rank(close - ts_max(close, 5)) ** ts_rank(corr(indneutralize(ts_mean(volume, 40)), low, 5), 3)) * -1", True),
    (91, "(ts_rank(decay_linear(decay_linear(corr(indneutralize(close), volume, 10), 16), 4), 5) - rank(decay_linear(corr(vwap, ts_mean(volume, 30), 4), 3))) * -1", True),
    (92, "min2(ts_rank(decay_linear(lt(((high + low) / 2) + close, low + open), 15), 19), ts_rank(decay_linear(corr(rank(low), rank(ts_mean(volume, 30)), 8), 7), 7))", False),
    (93, "ts_rank(decay_linear(corr(indneutralize(vwap), ts_mean(volume, 81), 17), 20), 8) / rank(decay_linear(delta((close * 0.524434) + (vwap * (1 - 0.524434)), 3), 16))", True),
    (94, "(rank(vwap - ts_min(vwap, 12)) ** ts_rank(corr(ts_rank(vwap, 20), ts_rank(ts_mean(volume, 60), 4), 18), 3)) * -1", False),
    (95, "lt(rank(open - ts_min(open, 12)), ts_rank(rank(corr(ts_sum((high + low) / 2, 19), ts_sum(ts_mean(volume, 40), 19), 13)) ** 5, 12))", False),
    (96, "max2(ts_rank(decay_linear(corr(rank(vwap), rank(volume), 4), 4), 8), ts_rank(decay_linear(ts_argmax(corr(ts_rank(close, 7), ts_rank(ts_mean(volume, 60), 4), 4), 13), 14), 13)) * -1", False),
    (97, "(rank(decay_linear(delta(indneutralize((low * 0.721001) + (vwap * (1 - 0.721001))), 3), 20)) - ts_rank(decay_linear(ts_rank(corr(ts_rank(low, 8), ts_rank(ts_mean(volume, 60), 17), 5), 19), 16), 7)) * -1", True),
    (98, "rank(decay_linear(corr(vwap, ts_sum(ts_mean(volume, 5), 26), 5), 7)) - rank(decay_linear(ts_rank(ts_argmin(corr(rank(open), rank(ts_mean(volume, 15)), 21), 9), 7), 8))", False),
    (99, "lt(rank(corr(ts_sum((high + low) / 2, 20), ts_sum(ts_mean(volume, 60), 20), 9)), rank(corr(low, volume, 6))) * -1", False),
    (100, "0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank((((close - low) - (high - close)) / (high - low)) * volume))))) - scale(indneutralize(corr(close, rank(ts_mean(volume, 20)), 5) - rank(ts_argmin(close, 30))))) * (volume / ts_mean(volume, 20))))", True),
    (101, "(close - open) / ((high - low) + 0.001)", False),
]


def load_alpha101(panel, include_approx: bool = True) -> List[Alpha]:
    """Return the Alpha101 expressions that parse/evaluate on the given panel.

    include_approx=False drops the alphas that depend on the IndNeutralize approximation.
    """
    out = []
    for aid, expr, approx in ALPHA101:
        if approx and not include_approx:
            continue
        try:
            a = validate(expr, panel)
            a.rationale = f"WorldQuant Alpha#{aid}" + (" (IndNeutralize approx)" if approx else "")
            a.meta = {"source": "alpha101", "id": aid, "approx": approx}
            out.append(a)
        except AlphaError:
            continue
    return out


def coverage_report(panel) -> dict:
    """Diagnostic: how many of the 101 parse + evaluate on this panel."""
    ok, failed = [], []
    for aid, expr, _ in ALPHA101:
        try:
            validate(expr, panel)
            ok.append(aid)
        except AlphaError as e:
            failed.append((aid, str(e)))
    return {"defined": len(ALPHA101), "ok": ok, "failed": failed,
            "skipped": SKIPPED}
