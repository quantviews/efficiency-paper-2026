# Data and code for "More Retail, No Less Efficiency"

Replication package for a study of short-horizon efficiency in Russian equities
after 2022 (Russian Journal of Money and Finance). [Русская версия](README.md).

Repository: <https://github.com/quantviews/efficiency-paper-2026>.

Archived version 1.0.0 on Zenodo: [10.5281/zenodo.23387914](https://doi.org/10.5281/zenodo.23387914).
Use this version-specific DOI when citing the data.

One script, no setup:

```bash
pip install numpy pandas
python replicate.py --check
```

## What is here, and what is not

The raw input is the complete order and trade log of the Moscow Exchange for
2021–2025 across 60 stocks: roughly 2.4 TB of archives. It cannot be
redistributed; access is governed by the exchange's terms. Between that raw log
and every conclusion in the paper, however, there is one narrow layer - a
**ticker-day panel of 5,732 rows** - and it is here in full.

| step | where | reproduced |
|:--|:--|:--|
| order log → book → snapshots on a fixed grid | with the authors | no, data are licensed |
| snapshots → ticker-day measures | `replicate.py --example` | on a synthetic day |
| measures → tables, intervals, regressions, conclusions | `replicate.py` | in full |

The third row is most of the paper: period means, before/after differences,
confidence intervals from a block bootstrap over dates, stock fixed-effects
regressions, liquidity-tier and sector breakdowns, the post-period trend, the
trading test and the structural-break tests.

The second row is checked on a synthetic day with **known** properties: the
quote midpoint follows a random walk, the spread is constant, and the side of
each trade is drawn independently. The estimators have to recover those
properties: VR(5) of the midpoint near one, negative first-order autocorrelation
of trade prices, a Roll estimate close to the imposed spread, and VR(5) of the
micro-price below one - queue noise that the midpoint does not carry. Real book
snapshots are not included: they are derived from exchange data.

## Files

### Data

| file | rows | contents |
|:--|--:|:--|
| `data/panel_5s.csv` | 5,732 | main panel: ticker-day measures, 5 s grid |
| `data/panel_1s.csv` | 5,732 | same on a 1 s grid (Section 7.1) |
| `data/panel_wholeday.csv` | 5,732 | same over the whole trading day (Section 7.4) |
| `data/price_variable.csv` | 357 | variance decomposition of the micro-price (Table 3) |
| `data/net_edge_q15.csv` | 5,432 | trading test, 1.5-sigma entry threshold (Section 6.5) |
| `data/net_edge_q10.csv`, `q20.csv` | 5,432 | the same at 1.0 and 2.0 sigma |
| `data/net_edge_scan.csv` | 5,732 | sweep over the entry threshold |
| `data/ofi_horizon.csv` | 6 | OFI R² by forecast horizon (Table 7) |
| `data/universe.csv` | 60 | liquidity rank, tier, sector |
| `data/example_snapshots.csv` | 6,240 | snapshot format example, synthetic |
| `data/figure_data.xlsx` | — | data behind every figure, one sheet per figure |
| `data/reference_numbers.json` | 630 | published values, for checking |
| `data/efficiency.yaml` | — | experiment definition: universe, periods, seed |
| `data/manifest_*.json` | — | run record: config hash, code version, omissions |

### Panel columns

`ticker`, `date`, `period` identify the stock, the trading day and the sample
period; `n_obs` is the number of book snapshots the day is computed from.

| column | measures |
|:--|:--|
| `vr5_mid`, `vr30_mid` | variance ratio of the quote midpoint at lags 5 and 30 steps |
| `vr5`, `vr30` | the same on the volume-weighted quote midpoint (micro-price) |
| `vr5_z`, `vr30_z` | heteroskedasticity-robust z (Lo and MacKinlay, 1988, M2) |
| `vr25s`, `vr25s_mid` | variance ratio at a fixed 25 s horizon |
| `ac1_mid`, `ac1_micro` | first-order autocorrelation of increments |
| `ac1_trade` | the same for trade prices, where bid-ask bounce lives |
| `roll_spread_bps` | effective spread, Roll (1984); empty when undefined |
| `quoted_spread_bps` | median quoted spread over the day, bps |
| `ofi_r2_60s`, `ofi_r2_300s` | share of future return variance explained by order-flow imbalance |
| `realized_vol_bps` | realised volatility over the day, bps |
| `turnover` | turnover, lots |

An empty cell means "not defined", not zero. This matters for
`roll_spread_bps`: substituting zero would bias the average spread downwards.

### Code

- `replicate.py` - the whole computation. Requires `numpy` and `pandas`, plus
  `matplotlib` for `--figures`. It does not import the authors' research
  package: it is an independent implementation, which is what makes the
  agreement of the numbers informative.

```bash
python replicate.py             # Tables 2-7 of the paper
python replicate.py --check     # compare against published numbers
python replicate.py --example   # measures computed from book snapshots
python replicate.py --figures   # plus figures
```

## On confidence intervals

Point estimates are deterministic and must match the published values digit for
digit. Interval bounds and p-values are random: the bootstrap here is seeded per
quantity rather than drawn from one stream across the whole report, as in the
original computation. `--check` therefore compares point estimates exactly and
p-values within four bootstrap standard errors: at p near 0.7 that noise is
three times what it is at p = 0.05, so a round constant would be either too
loose or too strict. Had the conclusions depended on a particular sequence of
draws, this is where it would show.

## How the bootstrap works

**Whole dates** are resampled, not individual ticker-days. Observations within a
day share a common market shock; resampling rows would destroy that dependence
and produce intervals that are too narrow. The sample has 105 dates against
5,732 rows, so the difference matters. B = 4000 replications, seed 12345; both
are in `data/efficiency.yaml`.

The lower bound on a bootstrap p-value is 1/(B+1): the procedure cannot return
zero, and printing "p = 0.000" would suggest an impossible event.

## Integrity

```bash
sha256sum -c CHECKSUMS.sha256
```

`VERSION.json` records the hash of the experiment configuration and the code
version the package was built with.

## Licences

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-data`), the licence under which
the journal publishes its material. Please cite the paper (`CITATION.cff`).

## Outside the package

Checksums of the raw order-log archives are not published. Anyone who obtains
the same data from the Moscow Exchange can follow the entire pipeline from
scratch using Sections 4 and 5 of the paper together with
`data/efficiency.yaml`, but byte-level identity of the source files is not
guaranteed here.
