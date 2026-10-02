#!/usr/bin/env python3
"""Воспроизведение таблиц статьи из опубликованных данных.

Скрипт самодостаточен: ему нужны только numpy и pandas и каталог `data/` рядом
с ним. Кода исследовательского пакета он не использует - это независимая
реализация статистической части, и совпадение чисел поэтому что-то значит.

    python replicate.py             # таблицы 2-7 статьи
    python replicate.py --check     # сверка с опубликованными значениями
    python replicate.py --example   # расчёт показателей из снимков книги
    python replicate.py --figures   # + рисунки (нужен matplotlib)

Что воспроизводится и что нет
-----------------------------
Панель `panel_5s.csv` - это показатели, уже посчитанные по журналу заявок:
одна строка на тикер-день. Скрипт повторяет ВСЁ, что делается дальше: средние
по периодам, разности «до/после», доверительные интервалы блочным бутстрапом
по датам, регрессии с фиксированными эффектами бумаг, разбивку по группам
ликвидности и секторам, тренд внутри пост-периода, торговую проверку.

Чего он НЕ проверяет на реальных данных: сам расчёт показателей из книги
заявок. Исходный журнал заявок Московской биржи распространять нельзя. Чтобы
этот шаг не оставался словами, `--example` считает те же показатели из
`data/example_snapshots.csv` - синтетического дня с заданными свойствами,
в том же формате полей, что и настоящие снимки.

Доверительные интервалы
-----------------------
Бутстрап здесь запускается со СВОИМ генератором на каждую величину
(`default_rng(SEED)` перед каждой оценкой), а не одним потоком случайных чисел
на весь отчёт, как в исходном расчёте. Точечные оценки от этого не зависят и
обязаны совпадать с опубликованными знак в знак. Границы интервалов и
p-значения - случайные величины, и совпадают лишь в пределах бутстрап-ошибки
(порядка 0,001 при B = 4000). `--check` учитывает это различие: точечные
оценки сверяются строго, интервалы и p - с допуском.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SEED = 12345
BOOT = 4000
#: периоды в хронологическом порядке; первое полугодие 2022 в выборку не входит
ORDER = ["2021", "2022-H2", "2023", "2024", "2025"]
PRE = "2021"
POST = ["2023", "2024", "2025"]

#: |VR(q) − 1|: расстояние до случайного блуждания, в обе стороны одинаково
ABS1 = lambda v: np.abs(v - 1.0)


# --------------------------------------------------------------------------
# данные
# --------------------------------------------------------------------------

def load(name: str) -> pd.DataFrame:
    p = DATA / f"{name}.csv"
    if not p.exists():
        raise SystemExit(f"нет файла {p}")
    df = pd.read_csv(p)
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def universe() -> pd.DataFrame:
    """Универсум с рангом ликвидности, группой и сектором."""
    return load("universe")


def attach(panel: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    return panel.merge(uni[["ticker", "tier", "sector"]], on="ticker", how="left")


# --------------------------------------------------------------------------
# статистика
# --------------------------------------------------------------------------

def _by_date(df: pd.DataFrame, key: str, tr=None):
    """Суммы и размеры метрики по датам - основа блочного бутстрапа.

    Ресемплятся ДАТЫ целиком, а не отдельные тикер-дни: наблюдения одного дня
    несут общий рыночный шок, и ресемплинг строк давал бы слишком узкие
    интервалы. Дат всего 105 при 5 732 строках, разница существенна.
    """
    v = pd.to_numeric(df[key], errors="coerce")
    ok = np.isfinite(v)
    v, d = v[ok].to_numpy(float), df.loc[ok, "date"].to_numpy()
    if tr is not None:
        v = tr(v)
    order = np.argsort(d, kind="stable")
    v, d = v[order], d[order]
    _, start = np.unique(d, return_index=True)
    sums = np.add.reduceat(v, start) if v.size else np.array([])
    cnts = np.diff(np.append(start, v.size)).astype(float) if v.size else np.array([])
    return sums, cnts, v


def boot_mean_ci(df: pd.DataFrame, key: str, tr=None, b: int = BOOT):
    """Среднее и 95%-интервал блочным бутстрапом по датам."""
    sums, cnts, allv = _by_date(df, key, tr)
    if allv.size < 3 or sums.size < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, sums.size, (b, sums.size))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(allv.mean()), float(lo), float(hi)


def boot_diff(pre: pd.DataFrame, post: pd.DataFrame, key: str, tr=None, b: int = BOOT):
    """Разность средних (post − pre), 95%-интервал и двусторонний p.

    p ограничен снизу величиной 1/(B+1): бутстрап не может дать «ноль».
    """
    sa, ca, va = _by_date(pre, key, tr)
    sb, cb, vb = _by_date(post, key, tr)
    if min(va.size, vb.size, sa.size, sb.size) < 3:
        return float("nan"), (float("nan"), float("nan")), float("nan")
    rng = np.random.default_rng(SEED)
    ia = rng.integers(0, sa.size, (b, sa.size))
    ib = rng.integers(0, sb.size, (b, sb.size))
    da = sb[ib].sum(1) / cb[ib].sum(1) - sa[ia].sum(1) / ca[ia].sum(1)
    lo, hi = np.percentile(da, [2.5, 97.5])
    p = max(2 * min((da > 0).mean(), (da < 0).mean()), 1.0 / (b + 1))
    return float(vb.mean() - va.mean()), (float(lo), float(hi)), float(min(p, 1.0))


def _design(df: pd.DataFrame, key: str, tr, with_vol: bool):
    """Матрица плана для `metric ~ post + C(ticker) [+ волатильность]`.

    Переходное полугодие 2022-H2 исключается: иначе оно попало бы в базу
    (post = 0) вместе с 2021 годом, и коэффициент сравнивал бы пост-период
    с «2021 плюс кризис», а не с 2021 годом, как сырая разность.
    """
    d = df[df["period"] != "2022-H2"].copy()
    y = pd.to_numeric(d[key], errors="coerce")
    ok = np.isfinite(y)
    if with_vol:
        ok &= np.isfinite(pd.to_numeric(d["realized_vol_bps"], errors="coerce"))
    d, y = d[ok], y[ok].to_numpy(float)
    if tr is not None:
        y = tr(y)
    dummies = pd.get_dummies(d["ticker"], dtype=float)
    cols = [(d["period"] != PRE).astype(float).rename("post"), dummies]
    if with_vol:
        cols.append(pd.to_numeric(d["realized_vol_bps"], errors="coerce").astype(float))
    x = pd.concat(cols, axis=1).to_numpy(float)
    return y, x, d["date"].to_numpy()


def fe_post(df: pd.DataFrame, key: str, tr=None, with_vol: bool = False, b: int = BOOT):
    """Коэффициент при post и его интервал; бутстрап ресемплит даты целиком."""
    y, x, dates = _design(df, key, tr, with_vol)
    if y.size < 10 or len(set(dates)) < 3:
        return float("nan"), (float("nan"), float("nan")), float("nan")
    beta = float(np.linalg.lstsq(x, y, rcond=None)[0][0])
    uniq = np.unique(dates)
    groups = [np.flatnonzero(dates == u) for u in uniq]
    rng = np.random.default_rng(SEED)
    coefs = np.empty(b)
    for i in range(b):
        sel = np.concatenate([groups[j] for j in rng.integers(0, uniq.size, uniq.size)])
        try:
            coefs[i] = np.linalg.lstsq(x[sel], y[sel], rcond=None)[0][0]
        except np.linalg.LinAlgError:
            coefs[i] = np.nan
    coefs = coefs[np.isfinite(coefs)]
    if coefs.size < 10:
        return beta, (float("nan"), float("nan")), float("nan")
    lo, hi = np.percentile(coefs, [2.5, 97.5])
    p = max(2 * min((coefs > 0).mean(), (coefs < 0).mean()), 1.0 / (b + 1))
    return beta, (float(lo), float(hi)), float(min(p, 1.0))


def trend_slope(df: pd.DataFrame, key: str, tr=None, b: int = BOOT):
    """Наклон по годам пост-периода (2023 = 0, 2024 = 1, 2025 = 2)."""
    xmap = {"2023": 0.0, "2024": 1.0, "2025": 2.0}
    d = df[df["period"].isin(xmap)].copy()
    y = pd.to_numeric(d[key], errors="coerce")
    ok = np.isfinite(y)
    d, y = d[ok], y[ok].to_numpy(float)
    if tr is not None:
        y = tr(y)
    x = d["period"].map(xmap).to_numpy(float)
    dates = d["date"].to_numpy()
    if y.size < 10:
        return float("nan"), (float("nan"), float("nan")), float("nan")

    def fit(yy, xx):
        return np.linalg.lstsq(np.vstack([xx, np.ones_like(xx)]).T, yy, rcond=None)[0][0]

    uniq = np.unique(dates)
    groups = [np.flatnonzero(dates == u) for u in uniq]
    rng = np.random.default_rng(SEED)
    sl = np.empty(b)
    for i in range(b):
        sel = np.concatenate([groups[j] for j in rng.integers(0, uniq.size, uniq.size)])
        sl[i] = fit(y[sel], x[sel])
    lo, hi = np.percentile(sl, [2.5, 97.5])
    p = max(2 * min((sl > 0).mean(), (sl < 0).mean()), 1.0 / (b + 1))
    return float(fit(y, x)), (float(lo), float(hi)), float(min(p, 1.0))


# --------------------------------------------------------------------------
# тесты структурного сдвига
# --------------------------------------------------------------------------

def monthly(df: pd.DataFrame, key: str, tr=None):
    """Месячный ряд среднего значения метрики: (год, месяц) -> среднее."""
    d = df.copy()
    v = pd.to_numeric(d[key], errors="coerce")
    ok = np.isfinite(v)
    d, v = d[ok], v[ok].to_numpy(float)
    if tr is not None:
        v = tr(v)
    d = d.assign(_v=v, _ym=[(x.year, x.month) for x in d["date"]])
    g = d.groupby("_ym")["_v"].mean().sort_index()
    return list(g.index), g.to_numpy(float)


def qlr(ys: np.ndarray, trim: float = 0.15) -> tuple[float, int]:
    """sup-Wald (Andrews, 1993) для сдвига среднего в неизвестной точке.

    Нулевая гипотеза - разрыва НЕТ. Возвращает максимум статистики и индекс,
    на котором он достигнут.
    """
    n = ys.size
    lo, hi = int(n * trim), int(n * (1 - trim))
    best, at = -np.inf, -1
    for k in range(max(lo, 2), min(hi, n - 2)):
        a, b_ = ys[:k], ys[k:]
        s2 = ((a - a.mean()) ** 2).sum() + ((b_ - b_.mean()) ** 2).sum()
        s2 /= max(n - 2, 1)
        if s2 <= 0:
            continue
        f = (a.mean() - b_.mean()) ** 2 / (s2 * (1 / a.size + 1 / b_.size))
        if f > best:
            best, at = f, k
    return float(best), at


def bai_perron(ys: np.ndarray, kmax: int = 3, min_seg: int = 6):
    """Число и положение разрывов среднего: динамическое программирование + BIC."""
    n = ys.size
    cs, cs2 = np.concatenate([[0], np.cumsum(ys)]), np.concatenate([[0], np.cumsum(ys ** 2)])

    def rss(i: int, j: int) -> float:
        m = j - i
        if m <= 0:
            return np.inf
        s, s2 = cs[j] - cs[i], cs2[j] - cs2[i]
        return float(s2 - s * s / m)

    best_k, best_bic, best_brk = 0, np.inf, []
    for k in range(kmax + 1):
        # dp[k][j] - минимальная RSS для первых j точек при k разрывах
        dp = np.full((k + 1, n + 1), np.inf)
        arg = np.zeros((k + 1, n + 1), int)
        for j in range(min_seg, n + 1):
            dp[0, j] = rss(0, j)
        for m in range(1, k + 1):
            for j in range(min_seg * (m + 1), n + 1):
                for i in range(min_seg * m, j - min_seg + 1):
                    val = dp[m - 1, i] + rss(i, j)
                    if val < dp[m, j]:
                        dp[m, j], arg[m, j] = val, i
        r = dp[k, n]
        if not np.isfinite(r) or r <= 0:
            continue
        bic = n * np.log(r / n) + (2 * k + 1) * np.log(n)
        if bic < best_bic:
            brk, j = [], n
            for m in range(k, 0, -1):
                j = arg[m, j]
                brk.append(j)
            best_k, best_bic, best_brk = k, bic, sorted(brk)
    return best_k, best_brk


# --------------------------------------------------------------------------
# таблицы
# --------------------------------------------------------------------------

METRICS = [
    ("vr5_mid", "|VR(5)−1| средняя котировка", 3, ABS1),
    ("vr30_mid", "|VR(30)−1| средняя котировка", 3, ABS1),
    ("ac1_mid", "автокорр r(средняя котировка)", 4, None),
    ("ac1_trade", "автокорр r(цена сделки)", 4, None),
    ("quoted_spread_bps", "котируемый спред, б.п.", 2, None),
    ("roll_spread_bps", "Roll-спред, б.п.", 2, None),
    ("ofi_r2_60s", "OFI R² (60 с)", 4, None),
    ("realized_vol_bps", "реализ. волатильность, б.п.", 0, None),
    ("vr5", "|VR(5)−1| микроцена", 3, ABS1),
]


def _f(x: float, nd: int) -> str:
    return "н/д" if not np.isfinite(x) else f"{x:.{nd}f}"


def _p(p: float) -> str:
    return "н/д" if not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"= {p:.3f}")


def table_periods(df: pd.DataFrame) -> str:
    periods = [p for p in ORDER if (df["period"] == p).any()]
    out = ["", "Таблица 2. Средние по периодам (сетка 5 с)", ""]
    out.append("метрика".ljust(34) + "".join(p.rjust(14) for p in periods))
    for key, label, nd, tr in METRICS:
        cells = []
        for p in periods:
            m, _, _ = boot_mean_ci(df[df["period"] == p], key, tr)
            cells.append(_f(m, nd).rjust(14))
        out.append(label.ljust(34) + "".join(cells))
    z = [np.abs(pd.to_numeric(df.loc[df["period"] == p, "vr5_z"], errors="coerce")) > 1.96
         for p in periods]
    out.append("доля дней |z(VR5)|>1,96".ljust(34)
               + "".join(f"{v.mean()*100:.0f}%".rjust(14) for v in z))
    return "\n".join(out)


def table_prepost(df: pd.DataFrame) -> str:
    pre = df[df["period"] == PRE]
    post = df[df["period"].isin(POST)]
    out = ["", "Таблица 4. 2021 против 2023-2025 (пул)", "",
           "метрика".ljust(34) + "Δ".rjust(10) + "95%-интервал".rjust(24) + "  p"]
    for key, label, nd, tr in METRICS:
        est, (lo, hi), p = boot_diff(pre, post, key, tr)
        out.append(label.ljust(34) + f"{est:+.4f}".rjust(10)
                   + f"[{lo:+.4f}; {hi:+.4f}]".rjust(24) + "  p " + _p(p))
    return "\n".join(out)


def table_fe(df: pd.DataFrame) -> str:
    out = ["", "Таблица 4 (продолжение). Коэффициент post с фиксированными эффектами", "",
           "метрика".ljust(34) + "post".rjust(10) + "95%-интервал".rjust(24)
           + "  p".ljust(12) + "post + волат.".rjust(14)]
    for key, label, tr in [("vr5_mid", "|VR(5)−1| средняя котировка", ABS1),
                           ("vr30_mid", "|VR(30)−1| средняя котировка", ABS1),
                           ("quoted_spread_bps", "котируемый спред", None),
                           ("ofi_r2_60s", "OFI R² (60 с)", None)]:
        b0, (lo, hi), p0 = fe_post(df, key, tr, with_vol=False)
        b1, _, _ = fe_post(df, key, tr, with_vol=True)
        out.append(label.ljust(34) + f"{b0:+.4f}".rjust(10)
                   + f"[{lo:+.4f}; {hi:+.4f}]".rjust(24)
                   + ("  p " + _p(p0)).ljust(12) + f"{b1:+.4f}".rjust(14))
    return "\n".join(out)


def table_tiers(df: pd.DataFrame) -> str:
    out = ["", "Таблица 5. Группы ликвидности", "",
           "группа".ljust(20) + "Δ|VR(5)−1|".rjust(12) + "  p".ljust(12)
           + "FE-post".rjust(10) + "Δ спред".rjust(10) + "Δ OFI R²".rjust(12)]
    for t in df["tier"].dropna().unique():
        d = df[df["tier"] == t]
        pre, post = d[d["period"] == PRE], d[d["period"].isin(POST)]
        dv, _, pv = boot_diff(pre, post, "vr5_mid", ABS1)
        fe, _, _ = fe_post(d, "vr5_mid", ABS1, with_vol=True)
        ds, _, _ = boot_diff(pre, post, "quoted_spread_bps")
        do, _, _ = boot_diff(pre, post, "ofi_r2_60s")
        out.append(f"{t} ({d['ticker'].nunique()})".ljust(20)
                   + f"{dv:+.4f}".rjust(12) + ("  p " + _p(pv)).ljust(12)
                   + f"{fe:+.4f}".rjust(10) + f"{ds:+.2f}".rjust(10)
                   + f"{do:+.5f}".rjust(12))
    return "\n".join(out)


def table_decomposition() -> str:
    d = load("price_variable")
    out = ["", "Таблица 3. Разложение дисперсии приращений микроцены", "",
           "период".rjust(14) + "доля очереди".rjust(14) + "доля средней".rjust(14)
           + "ковариация".rjust(14) + "VR(5) очереди".rjust(16) + "спред, тиков".rjust(14)]
    for p in [x for x in ORDER if (d["period"] == x).any()]:
        s = d[d["period"] == p]
        out.append(p.rjust(14)
                   + _f(s["var_share_queue"].mean(), 3).rjust(14)
                   + _f(s["var_share_mid"].mean(), 3).rjust(14)
                   + _f(s["cov_share"].mean(), 3).rjust(14)
                   + _f(s["vr5_queue"].mean(), 3).rjust(16)
                   + _f(s["spread_ticks"].mean(), 2).rjust(14))
    return "\n".join(out)


def table_net_edge() -> str:
    d = load("net_edge_q15")
    # Столбцы `*_bps` - суммы за день. Чтобы сравнить с издержками round-trip,
    # делим сумму по периоду на общее число сделок: получается величина НА СДЕЛКУ.
    # Простое среднее по тикер-дням дало бы другое число - день с одной сделкой
    # весил бы столько же, сколько день с четырьмя сотнями.
    out = ["", "Таблица 6. Торговая проверка правила по дисбалансу потока заявок",
           "", "на одну сделку, базисные пункты; порог входа 1,5 сигмы", "",
           "период".rjust(14) + "сделок".rjust(10) + "валовая".rjust(12)
           + "издержки".rjust(12) + "нетто".rjust(12)]
    for p in [x for x in ORDER if (d["period"] == x).any()]:
        s = d[d["period"] == p]
        n = s["n_trades"].sum()
        if not n:
            continue
        out.append(p.rjust(14) + f"{int(round(n))}".rjust(10)
                   + _f(s["gross_bps"].sum() / n, 2).rjust(12)
                   + _f(s["cost_bps"].sum() / n, 2).rjust(12)
                   + _f(s["net_bps"].sum() / n, 2).rjust(12))
    return "\n".join(out)


def table_ofi_horizon() -> str:
    d = load("ofi_horizon")
    out = ["", "Таблица 7. OFI R² по горизонтам прогноза", "",
           "горизонт, с".rjust(14) + "2021".rjust(12) + "2024".rjust(12)]
    for _, r in d.iterrows():
        out.append(f"{int(r['horizon_s'])}".rjust(14)
                   + _f(r["ofi_r2_2021"], 4).rjust(12) + _f(r["ofi_r2_2024"], 4).rjust(12))
    return "\n".join(out)


def section_breaks(df: pd.DataFrame) -> str:
    months, ys = monthly(df, "vr5_mid", ABS1)
    f, at = qlr(ys)
    k, brk = bai_perron(ys)
    out = ["", "Раздел 7.3. Тесты структурного сдвига (|VR(5)−1| средней котировки)", "",
           f"месячных наблюдений: {len(months)}",
           f"sup-Wald (Andrews, 1993): {f:.2f}"
           + (f", максимум на {months[at][0]}-{months[at][1]:02d}" if at > 0 else ""),
           f"Bai-Perron (BIC, kmax = 3): разрывов {k}"
           + (", в " + ", ".join(f"{months[i][0]}-{months[i][1]:02d}" for i in brk)
              if brk else "")]
    return "\n".join(out)


# --------------------------------------------------------------------------
# сверка с опубликованными значениями
# --------------------------------------------------------------------------

def _ru(x: float, nd: int) -> str:
    """Формат чисел статьи: десятичная запятая и типографский минус."""
    return "н/д" if not np.isfinite(x) else f"{x:.{nd}f}".replace("-", "−").replace(".", ",")


def check(df: pd.DataFrame) -> int:
    """Сверить пересчитанные величины с опубликованными. Возвращает число расхождений.

    Точечные оценки сверяются строго, в том формате, в каком они напечатаны
    в статье. Границы интервалов и p-значения - случайные величины бутстрапа,
    для них проверяется совпадение с допуском.
    """
    ref_path = DATA / "reference_numbers.json"
    if not ref_path.exists():
        print("нет data/reference_numbers.json - сверять не с чем")
        return 0
    ref = json.loads(ref_path.read_text(encoding="utf-8"))

    exact: dict[str, str] = {}
    tol: dict[str, tuple[float, float]] = {}

    periods = [p for p in ORDER if (df["period"] == p).any()]
    keymap = {"2022-H2": "2022H2"}
    for key, _label, nd, tr in [
            ("vr5_mid", "", 3, ABS1), ("vr30_mid", "", 3, ABS1),
            ("vr5", "", 3, ABS1), ("vr30", "", 3, ABS1)]:
        for p in periods:
            m, _, _ = boot_mean_ci(df[df["period"] == p], key, tr)
            name = "abs_" + key
            exact[f"{name}_{keymap.get(p, p)}"] = _ru(m, nd)
    for key, nd in [("vr5", 3), ("vr30", 3), ("ac1_micro", 3), ("ac1_mid", 3),
                    ("ac1_trade", 3), ("roll_spread_bps", 2), ("quoted_spread_bps", 2),
                    ("ofi_r2_60s", 4), ("ofi_r2_300s", 4), ("realized_vol_bps", 0)]:
        for p in periods:
            m, _, _ = boot_mean_ci(df[df["period"] == p], key)
            exact[f"{key}_{keymap.get(p, p)}"] = _ru(m, nd)

    pre, post = df[df["period"] == PRE], df[df["period"].isin(POST)]
    diffs = [("d_absvr5mid", "vr5_mid", ABS1, 3), ("d_absvr30mid", "vr30_mid", ABS1, 3),
             ("d_absvr5", "vr5", ABS1, 3), ("d_absvr30", "vr30", ABS1, 3),
             ("d_qspread", "quoted_spread_bps", None, 2),
             ("d_roll", "roll_spread_bps", None, 2),
             ("d_ofi60", "ofi_r2_60s", None, 4), ("d_ofi300", "ofi_r2_300s", None, 4),
             ("d_ac1mid", "ac1_mid", None, 3), ("d_ac1micro", "ac1_micro", None, 3),
             ("d_ac1trade", "ac1_trade", None, 3), ("d_vol", "realized_vol_bps", None, 0)]
    for name, key, tr, nd in diffs:
        est, _ci, p = boot_diff(pre, post, key, tr)
        exact[name] = ("+" if est >= 0 else "−") + _ru(abs(est), nd)
        tol[name + "_p"] = (p, 0.02)

    for name, key, tr in [("fe_absvr5mid", "vr5_mid", ABS1),
                          ("fe_absvr30mid", "vr30_mid", ABS1),
                          ("fe_absvr5", "vr5", ABS1), ("fe_absvr30", "vr30", ABS1),
                          ("fe_ofi60", "ofi_r2_60s", None)]:
        b0, _ci, p0 = fe_post(df, key, tr, with_vol=False)
        nd = 4 if key.startswith("ofi") else 3
        exact[name] = ("+" if b0 >= 0 else "−") + _ru(abs(b0), nd)
        tol[name + "_p"] = (p0, 0.02)

    n_exact = n_bad = 0
    bad: list[str] = []
    for name, got in exact.items():
        if name not in ref:
            continue
        n_exact += 1
        if str(ref[name]).strip() != got:
            n_bad += 1
            bad.append(f"  {name}: опубликовано {ref[name]}, пересчитано {got}")
    n_tol = n_tol_bad = 0
    for name, (got, _eps) in tol.items():
        if name not in ref:
            continue
        n_tol += 1
        txt = str(ref[name]).replace("<", "").replace("=", "").replace(",", ".").strip()
        try:
            want = float(txt)
        except ValueError:
            continue
        # p = 2·min(доля), доля ~ Bin(B, ·): стандартная ошибка одной оценки
        # равна 2·√(f(1−f)/B). Допуск в четыре такие ошибки, а не круглая
        # константа: при p около 0,7 бутстрап-шум втрое больше, чем при p = 0,05.
        f = min(max(want / 2, 1e-4), 0.5)
        eps = max(0.01, 4 * 2 * np.sqrt(f * (1 - f) / BOOT))
        if abs(want - got) > eps and not (want < 0.001 and got < 0.01):
            n_tol_bad += 1
            bad.append(f"  {name}: опубликовано {ref[name]}, пересчитано {got:.4f} "
                       f"(допуск {eps:.3f})")

    print("\nСверка с опубликованными значениями")
    print(f"  точечные оценки: {n_exact} сверено, расхождений {n_bad}")
    print(f"  p-значения (допуск 4 бутстрап-ошибки): {n_tol} сверено, "
          f"расхождений {n_tol_bad}")
    if bad:
        print("\n".join(bad))
    else:
        print("  все сверенные величины совпали")
    return n_bad + n_tol_bad


# --------------------------------------------------------------------------
# показатели из снимков книги (пример формата)
# --------------------------------------------------------------------------

def variance_ratio(r: np.ndarray, q: int) -> tuple[float, float]:
    """VR(q) и робастный к гетероскедастичности z (Lo, MacKinlay, 1988, версия M2).

    VR = Var(q-шаговой доходности) / (q · Var(1-шаговой)); VR = 1 - случайное
    блуждание. Обычный статистик на финансовых рядах даёт ложную значимость,
    поэтому используется робастная версия.
    """
    r = r[np.isfinite(r)]
    n = r.size
    if n < q * 3 or q < 2:
        return float("nan"), float("nan")
    mu = r.mean()
    dev = r - mu
    var1 = (dev @ dev) / (n - 1)
    if var1 <= 0:
        return float("nan"), float("nan")
    csum = np.concatenate([[0.0], np.cumsum(r)])
    qsum = csum[q:] - csum[:-q]
    var_q = ((qsum - q * mu) ** 2).sum() / (q * (n - q + 1) * (1.0 - q / n))
    vr = var_q / var1
    dev2 = dev ** 2
    denom = dev2.sum() ** 2
    theta = sum((2.0 * (q - j) / q) ** 2 * (dev2[j:] @ dev2[:-j]) / denom
                for j in range(1, q))
    z = (vr - 1.0) / np.sqrt(theta) if theta > 0 else float("nan")
    return float(vr), float(z)


def autocorr(r: np.ndarray, lag: int = 1) -> float:
    r = r[np.isfinite(r)]
    if r.size <= lag + 2 or r[:-lag].std() == 0 or r[lag:].std() == 0:
        return float("nan")
    return float(np.corrcoef(r[:-lag], r[lag:])[0, 1])


def roll_spread_bps(price: np.ndarray) -> float:
    """Эффективный спред по Roll (1984): s = 2·√(−Cov(Δp_t, Δp_{t−1})), б.п.

    При неотрицательной ковариации модель неприменима и оценка не определена:
    возвращается NaN, а не ноль. Ноль занижал бы средний спред тем сильнее,
    чем чаще встречаются такие дни.
    """
    p = price[np.isfinite(price)]
    if p.size < 100:
        return float("nan")
    dp = np.diff(p)
    if dp.size < 50 or dp.std() == 0:
        return float("nan")
    cov = np.cov(dp[1:], dp[:-1])[0, 1]
    if cov >= 0:
        return float("nan")
    return float(2.0 * np.sqrt(-cov) / p.mean() * 1e4)


def metrics_from_snapshots(s: pd.DataFrame, grid_ms: int = 5000) -> dict:
    """Показатели тикер-дня из снимков книги - тот шаг, которого нет в панели.

    Вход - ровно то, что видно в потоковом режиме: лучшие котировки с объёмами
    и цена последней сделки в интервале. Ничего из полного журнала заявок здесь
    не используется.
    """
    pb = s["bid_px_e6"].to_numpy(float)
    pa = s["ask_px_e6"].to_numpy(float)
    qb = s["bid_qty"].to_numpy(float)
    qa = s["ask_qty"].to_numpy(float)
    two_sided = (pb > 0) & (pa > 0)

    total = qb + qa
    micro = np.where(two_sided & (total > 0), (pb * qa + pa * qb) / total, np.nan)
    mid = np.where(two_sided, (pb + pa) / 2, np.nan)
    spread = np.where(two_sided, (pa - pb) / mid * 1e4, np.nan)

    def ret(p, lag=1):
        out = np.full(p.shape, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[lag:] = np.log(p[lag:] / p[:-lag]) * 1e4
        return out

    def fwd(p, h):
        out = np.full(p.shape, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:-h] = np.log(p[h:] / p[:-h]) * 1e4
        return out

    # OFI на лучшем уровне (Cont, Kukanov, Stoikov, 2014), накопленный за 60 с
    e_b = np.where(pb[1:] >= pb[:-1], qb[1:], 0) - np.where(pb[1:] <= pb[:-1], qb[:-1], 0)
    e_a = np.where(pa[1:] <= pa[:-1], qa[1:], 0) - np.where(pa[1:] >= pa[:-1], qa[:-1], 0)
    ofi = np.concatenate([[0.0], (e_b - e_a).astype(float)])
    step_s = grid_ms / 1000.0
    win = max(1, round(60 / step_s))
    if win > 1:
        cs = np.cumsum(ofi)
        roll = np.full(ofi.shape, np.nan)
        roll[win - 1:] = cs[win - 1:] - np.concatenate([[0.0], cs[:-win]])
        ofi = roll

    last = s["last_px_e6"].to_numpy(float)
    trade_px = np.where(last > 0, last, np.nan)
    trade_px = trade_px[np.isfinite(trade_px)]

    r_micro = ret(micro)
    r_mid = ret(mid)
    fr = fwd(micro, max(1, round(60 / step_s)))
    ok = np.isfinite(ofi) & np.isfinite(fr)
    r2 = (float(np.corrcoef(ofi[ok], fr[ok])[0, 1] ** 2)
          if ok.sum() >= 100 and ofi[ok].std() > 0 and fr[ok].std() > 0 else np.nan)

    vr5_mid, _ = variance_ratio(r_mid[np.isfinite(r_mid)], 5)
    vr5, z5 = variance_ratio(r_micro[np.isfinite(r_micro)], 5)
    return {
        "n_obs": int(np.isfinite(micro).sum()),
        "vr5_mid": vr5_mid, "vr5": vr5, "vr5_z": z5,
        "ac1_mid": autocorr(r_mid), "ac1_micro": autocorr(r_micro),
        "ac1_trade": autocorr(ret(trade_px)),
        "roll_spread_bps": roll_spread_bps(trade_px),
        "quoted_spread_bps": float(np.nanmedian(spread)),
        "ofi_r2_60s": r2,
        "realized_vol_bps": float(np.sqrt(np.nansum(r_micro ** 2))),
    }


def example() -> str:
    """Показать, как показатели получаются из снимков книги.

    `data/example_snapshots.csv` - синтетический тикер-день с ЗАДАННЫМИ
    свойствами: середина котировки идёт случайным блужданием, спред постоянен,
    сделки чередуются между лучшим спросом и предложением. Реальные котировки
    Московской биржи распространять нельзя, поэтому пример смоделирован; формат
    полей и способ расчёта - те же, что в основном прогоне.

    Проверяется не совпадение с панелью, а поведение оценок: на средней
    котировке VR(5) должен быть около единицы, автокорреляция цен сделок -
    заметно отрицательной (бид-аск отскок), оценка Ролла - близкой к заданному
    спреду, а микроцена - показывать возвратность, которой в средней нет.
    """
    p = DATA / "example_snapshots.csv"
    if not p.exists():
        return "\nнет data/example_snapshots.csv"
    s = pd.read_csv(p)
    meta_path = DATA / "example_snapshots.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    m = metrics_from_snapshots(s)

    out = ["", "Пример: показатели из снимков книги (синтетический день)", "",
           f"снимков: {len(s)}, шаг сетки: {meta.get('grid_ms', 5000)} мс"]
    if meta:
        out.append(f"заданный спред: {meta['spread_bps']:.2f} б.п.; "
                   f"волатильность шага: {meta['sigma_bps']:.1f} б.п.")
    out += ["",
            "показатель".ljust(30) + "получено".rjust(12) + "   ожидание",
            "VR(5) средней котировки".ljust(30) + _f(m["vr5_mid"], 3).rjust(12)
            + "   около 1 (случайное блуждание)",
            "VR(5) микроцены".ljust(30) + _f(m["vr5"], 3).rjust(12)
            + "   ниже 1 (возвратность от очереди)",
            "автокорр r(цена сделки)".ljust(30) + _f(m["ac1_trade"], 3).rjust(12)
            + "   отрицательна (бид-аск отскок)",
            "Roll-спред, б.п.".ljust(30) + _f(m["roll_spread_bps"], 2).rjust(12)
            + (f"   около {meta['spread_bps']:.2f} (заданный спред)" if meta else ""),
            "котируемый спред, б.п.".ljust(30) + _f(m["quoted_spread_bps"], 2).rjust(12)
            + (f"   {meta['spread_bps']:.2f} по построению" if meta else "")]
    return "\n".join(out)


# --------------------------------------------------------------------------
# рисунки
# --------------------------------------------------------------------------

def figures(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    periods = [p for p in ORDER if (df["period"] == p).any()]
    x = np.arange(len(periods))

    def series(key, tr):
        m, lo, hi = [], [], []
        for p in periods:
            a, b_, c = boot_mean_ci(df[df["period"] == p], key, tr)
            m.append(a); lo.append(a - b_); hi.append(c - a)
        return np.array(m), np.array([lo, hi])

    for fname, ylabel, specs, hline in [
            ("fig1_spread.png", "спред, б.п.",
             [("quoted_spread_bps", "котируемый", None),
              ("roll_spread_bps", "Roll (эффективный)", None)], None),
            ("fig2_vr.png", "|VR(5)−1|",
             [("vr5_mid", "средняя котировка", ABS1),
              ("vr5", "микроцена", ABS1)], 0.0)]:
        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        for key, lbl, tr in specs:
            m, err = series(key, tr)
            ax.errorbar(x, m, yerr=err, marker="o", capsize=3, label=lbl)
        if hline is not None:
            ax.axhline(hline, color="0.3", lw=0.8, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(periods)
        ax.set_ylabel(ylabel); ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=300)
        plt.close(fig)

    tiers = [t for t in df["tier"].dropna().unique()]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for t in tiers:
        ys = [ABS1(pd.to_numeric(
            df[(df["period"] == p) & (df["tier"] == t)]["vr5_mid"],
            errors="coerce").dropna().to_numpy()).mean() for p in periods]
        ax.plot(x, ys, marker="o", label=t)
    ax.axhline(0, color="0.3", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(periods)
    ax.set_ylabel("|VR(5)−1| средней котировки"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "fig3_tiers.png", dpi=300)
    plt.close(fig)
    print(f"рисунки -> {out_dir}")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="сверить с опубликованными числами")
    ap.add_argument("--example", action="store_true",
                    help="показать расчёт показателей из снимков книги")
    ap.add_argument("--figures", action="store_true", help="построить рисунки (matplotlib)")
    ap.add_argument("--data", type=Path, default=None, help="каталог с данными")
    args = ap.parse_args()

    global DATA
    if args.data:
        DATA = args.data

    uni = universe()
    df = attach(load("panel_5s"), uni)
    print(f"панель: {len(df)} тикер-дней, {df['ticker'].nunique()} бумаг, "
          f"{df['date'].nunique()} дат, периоды {sorted(df['period'].unique())}")

    print(table_periods(df))
    print(table_prepost(df))
    print(table_fe(df))
    print(table_tiers(df))
    print(table_decomposition())
    print(table_net_edge())
    print(table_ofi_horizon())
    print(section_breaks(df))

    if args.example:
        print(example())

    if args.figures:
        figures(df, HERE / "figures")

    if args.check:
        raise SystemExit(1 if check(df) else 0)


if __name__ == "__main__":
    main()
