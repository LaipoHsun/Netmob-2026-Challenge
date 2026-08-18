# Phase 1 到 Phase 4 完整研究總整理

本文件整合 `phase1_report.md`、`phase2_report.md`、`phase3_report.md`、`phase4_report.md` 與對應輸出目錄。它的目的不是重新做一次分析，而是清楚交代我在四個 phase 中做了什麼、每一步為什麼做、產出哪些檔案、得到哪些結論，以及哪些主張最後被保留或被否定。

## 一句話總結

整個專案最後形成的物理圖像是：Niteroi 公車系統的同步與 bunching 比較像「空間上會放大的服務不穩定性」，而不是一個清楚的平衡態臨界轉變。原始 Kuramoto order parameter `r` 會被車輛數 `N` 的有限樣本效應嚴重影響，所以必須做 same-N finite-size correction。修正後，需求量每車 `lambda = boardings / N` 與 bunching 有小但穩定的正相關，但 Phase 3 的預註冊 criticality gate 不支持 sharp critical transition。Phase 4 又進一步顯示，shared corridor 的同步訊號大多可由共同道路速度狀態解釋，不能直接宣稱有強的跨路線傳播耦合。

## 檔案與流程總覽

| Phase | 主要腳本 | 報告 | 主要輸出目錄 | 主要問題 |
| --- | --- | --- | --- | --- |
| Phase 1 | `scripts/phase1_kuramoto_pipeline.py` | `phase1_report.md` | `outputs/phase1/` | 建立最小可行的 physics/network observables：map matching、phase、headway、order parameter、demand |
| Phase 2 | `scripts/phase2_analysis.py` | `phase2_report.md` | `outputs/phase2/` | 擴大到更多日期與路線，修正 finite-size artifact，檢驗 weekday/weekend、需求、天氣、初步 corridor coupling |
| Phase 3 | `scripts/phase3_analysis.py` | `phase3_report.md` | `outputs/phase3/` | 用預先定義的 gate 區分 critical transition 與 smooth crossover，加入 dwell、spatial amplification、within-pair corridor coupling |
| Phase 4 | `scripts/phase4_analysis.py` | `phase4_report.md` | `outputs/phase4/` | 控制 common shock，判斷 shared corridor 訊號是否仍可解釋為真正耦合，並完成 final synthesis figures |

主要可重跑命令：

```bash
python scripts/phase1_kuramoto_pipeline.py --skip-full-discovery
python scripts/phase2_analysis.py
python scripts/phase3_analysis.py
python scripts/phase4_analysis.py
```

Phase 1 如果要重新掃描全月資料，可以移除 `--skip-full-discovery`。

## 共通資料基礎與重要限制

### 1. 本地資料狀況

一開始我先做資料 discovery。原本題目或外部環境可能提到的 `/mnt/user-data/uploads`、`/mnt/data`、`data_description.pdf` 在本地工作區不可用，所以 Phase 1 改用本地 README 與實際 CSV schema 來建立 pipeline。

實際資料覆蓋：

| 資料 | 覆蓋時間 | 筆數與檔案 |
| --- | --- | --- |
| Mobility telemetry | `2026-03-11 00:00:07` 到 `2026-03-31 23:37:32` | 13,850,467 rows，19 files |
| Ticketing | `2026-03-01 00:00:02-03:00` 到 `2026-03-31 23:59:57-03:00` | 6,790,681 rows，31 files |

Mobility 的實際欄位是：

```text
id, timestamp, tripId, lat, lng, heading, lineId, lineName, headsign, direction
```

這跟一開始可能預期的 `linha`、`nomeLinha`、`angle`、`sentido` 不同，所以 Phase 1 pipeline 有針對實際欄位重寫。

### 2. 不能做車輛層級 demand join

我檢查 mobility 的 `id` 與 ticketing 的 `vehicle_number`：

| 指標 | 數值 |
| --- | --- |
| Mobility distinct `id` | 527 |
| Ticket distinct `vehicle_number` | 556 |
| overlap | 0 |
| overlap rate | 0% |

這是一個非常關鍵的限制。因為兩邊車號完全對不上，所以不能把乘客刷卡需求精準分配到單一車輛。後續所有 demand 都是以 line-hour 或 line-time-bin 聚合後使用，再分配到該 line 的方向或車隊狀態。這也是為什麼 `lambda = boardings / N` 是路線小時層級的每車需求 proxy，不是每台車真實載客量。

### 3. 路線 geometry 與 map matching

我使用兩種 geometry：

1. `line_routes.json` 的靜態路線 geometry。
2. GTFS geometry fallback，尤其用來處理靜態 geometry 缺失或品質不好的路線。

所有 phase 都非常重視 map matching 品質。因為後續的相位 `phi`、headway crossing、spatial amplification 都建立在車輛沿路線的位置 `s_m` 上。如果 matching 殘差太大，物理量就不可信。

### 4. Kuramoto phase 的定義

每一個 `(lineId, direction)` 被視為一條經驗閉合路線。車輛沿路線的位置 `s` 被轉成相位：

```text
phi = 2 * pi * s / L
```

其中 `L` 是該路線方向的 route length。這讓同一路線方向上的車輛可以被表示成圓周上的 oscillator，進而計算 Kuramoto order parameter：

```text
r = |mean(exp(i * phi_j))|
```

`r` 越高代表車輛相位越集中，也就是越同步或越 bunching；`r` 越低代表車輛沿路線分布越均勻。

## Phase 1 做了什麼

### 目標

Phase 1 的目標是建立第一版完整 pipeline，從原始 GPS 與 ticketing 資料產生 physics/network observables。重點不是一次涵蓋全城市，而是先在少數日期與高需求路線上驗證：資料能否被 map match、相位能否穩定計算、headway 與 demand 能否整合。

### 使用範圍

Phase 1 聚焦於：

| 類別 | 範圍 |
| --- | --- |
| 日期 | `2026-03-11` 到 `2026-03-14` |
| 路線 | `49.2`, `45`, `49.1`, `48`, `35`, `62` |
| ticket alias | `62B` 對應到 telemetry 的 `62` |

這幾條線是早期高需求與可用 geometry 的代表路線。

### 主要處理

Phase 1 完成以下工作：

1. 掃描 mobility 與 ticketing 檔案，建立 dataset discovery。
2. 確認 mobility 與 ticketing 的車號無法 join。
3. 清理 out-of-service 或缺失 lat/lng 的 mobility records。
4. 使用 static route geometry 與 GTFS fallback 做 map matching。
5. 將每筆 GPS 投影到路線 arc-length `s_m`。
6. 用 `phi = 2*pi*s/L` 轉成相位。
7. 將軌跡重採樣到 30 秒 grid。
8. 避免跨大缺口插值，超過 3 分鐘的 gaps 會切開。
9. 避免路線跳躍污染軌跡，超過 50% route length 的 jumps 會切開。
10. 計算 10 分鐘 line-time-bin 的 order parameter、demand 與 headway observables。
11. 在 5%、25%、50%、75%、95% cross sections 計算 crossing events 與 headway。
12. 以 rolling 30 分鐘 headway CV 描述 irregularity。
13. 定義 bunching event 為 headway 小於 local median observed headway 的 25%。

### 資料品質結果

Phase 1 中，focus days/lines 的 out-of-service 或 missing lat/lng 清除量為 0。這代表 GPS 基本可用。

Map matching 後，最終品質沒有 poor flags。特別是 line `62` 一開始需要靠 GTFS exact geometry 修正，修正後各路線 p95 residual 都低於 150 m。

軌跡處理中有兩個重要保護：

| 保護 | 次數 |
| --- | ---: |
| 超過 3 分鐘的 GPS gap 被切開 | 3,620 |
| 超過 50% route length 的 route jump 被切開 | 4,203 |

這些處理避免把不連續的 GPS 點硬插值成不存在的車輛移動。

### Phase 1 主要輸出

| 檔案 | 內容 |
| --- | --- |
| `outputs/phase1/dataset_discovery.json` | 原始資料掃描摘要 |
| `outputs/phase1/filtering_log.json` | 清理與過濾紀錄 |
| `outputs/phase1/map_matching_quality.csv` | 每條線/方向的 map matching 品質 |
| `outputs/phase1/projected_observations_sample.csv` | 投影到路線後的 GPS sample |
| `outputs/phase1/resampled_phase_30s.csv` | 30 秒相位重採樣資料，約 464,067 rows |
| `outputs/phase1/trajectory_gap_log.csv` | gap 與 jump 切分紀錄 |
| `outputs/phase1/headway_crossing_events.csv` | cross-section crossing events |
| `outputs/phase1/headway_cv_10min.csv` | 10 分鐘 headway CV |
| `outputs/phase1/line_demand_10min.csv` | 10 分鐘 line demand |
| `outputs/phase1/phase1_line_timebin_observables.csv` | Phase 1 主表，5,802 rows |
| `outputs/phase1/kuramoto_order_10min.csv` | 10 分鐘 Kuramoto order parameter |

Phase 1 也輸出多張早期圖：

| 圖 | 檔案 |
| --- | --- |
| Marey busy vs quiet | `outputs/phase1/f1_marey_busy_vs_quiet.png` |
| Headway distribution/CV | `outputs/phase1/f2_headway_distribution_cv.png` |
| Order parameter vs demand | `outputs/phase1/f3_order_parameter_demand.png` |
| Cross-line scatter | `outputs/phase1/f4_demand_vs_order_scatter.png` |

### Phase 1 主要發現

Phase 1 的 raw `r` 在 weekend 反而偏高，即使 weekend demand 較低。這個現象一開始看起來像「低需求反而更同步」，但我判斷這很可能是 finite-size artifact：weekend 車輛數 `N` 較小，而 Kuramoto `r` 在小 `N` 下即使隨機分布也會偏高。

這個懷疑直接導向 Phase 2 的核心任務：建立 same-N finite-size correction。

## Phase 2 做了什麼

### 目標

Phase 2 的目標是把 Phase 1 的 pipeline 擴大到更多日期與路線，並正式解決 Phase 1 發現的 finite-size artifact。它把研究從「可行性」推進到「可比較的城市尺度統計」。

### 擴大範圍

Phase 2 使用 mobility 與 ticketing 都有重疊的日期：

```text
2026-03-11, 2026-03-12, 2026-03-13, 2026-03-14, 2026-03-15,
2026-03-16, 2026-03-17, 2026-03-20, 2026-03-21, 2026-03-22,
2026-03-23, 2026-03-24, 2026-03-25, 2026-03-26, 2026-03-27,
2026-03-28, 2026-03-29, 2026-03-30, 2026-03-31
```

最終選入的 top-ridership 且 map-matching 通過的 20 條路線是：

```text
30, 48, 33, 49.2, 46, 67, 49.1, 45, 36, OC3,
39A, 35, 62, 38A, OC2, 61, OC1, 31, 44, 34A
```

候選路線 `47` 與 `53` 因 map-matching 品質不達標被排除，改由 `44` 與 `34A` 補上。

### Bus-hour 品質控管

Phase 2 改以 bus-hour 作為主要活動單位，並排除不適合估計相位的車輛小時：

| 排除原因 | 排除 bus-hours |
| --- | ---: |
| Terminal layover | 3,920 |
| Sparse midpoint observations | 11,326 |
| High median residual | 948 |
| 最終保留 active bus-hours | 69,466 |

這些規則的目的，是只保留真正在線上移動、且 map-matching 可信的車輛小時。

### Phase 2 主表

Phase 2 主表是：

```text
outputs/phase2/phase2_hourly_observables.csv
```

它有 11,641 rows，其中 10,135 rows 的 `N >= 2`，可用於 corrected synchronization 分析。

### Finite-size correction

這是 Phase 2 最核心的方法改進。

因為 raw Kuramoto `r` 會隨車輛數 `N` 改變，即使車輛相位是完全隨機的，小 `N` 也會得到較高的 expected `r`。所以我對每個實際出現的 `N` 做 5,000 次 random phase null simulation，得到：

```text
E_null[r | N]
sd_null[r | N]
```

再定義：

```text
r_excess = (r1 - E_null[r | N]) / (1 - E_null[r | N])
z = (r1 - E_null[r | N]) / sd_null[r | N]
```

其中：

| 指標 | 意義 |
| --- | --- |
| `r1` | 一階 Kuramoto order parameter |
| `E_null[r | N]` | 同車輛數下完全隨機相位的期望 order |
| `r_excess` | 扣掉 finite-size baseline 後，距離完美同步的相對 excess |
| `z` | 以 null standard deviation 標準化後的偏離程度 |

`N = 1` 的 rows 不適合做 synchronization correction，所以從 corrected modeling 中排除。

### T1 weekday/weekend 結果

Phase 2 直接推翻 Phase 1 raw `r` 的表面現象。

| 指標 | weekend minus weekday | 95% CI | 解讀 |
| --- | ---: | --- | --- |
| raw `r1` | 0.047 | 0.039 到 0.056 | weekend 看起來更同步 |
| corrected `r_excess` | -0.067 | -0.085 到 -0.048 | 修正後 weekend 其實較不 bunching |
| corrected `z` | -0.441 | -0.486 到 -0.397 | same-N 標準化後也較低 |

結論：Phase 1 看到的 weekend raw `r` 偏高，主要是 finite-size artifact，不是 weekend 真的更同步。

### Demand/load 模型

由於車輛層級 ticket join 不可行，Phase 2 定義：

```text
lambda = boardings / N
```

這裡的 `boardings` 是 line-hour demand，`N` 是該 line-direction-hour 的 active buses。

Phase 2 用 manual numpy/scipy 實作 random-intercept style model，原因是當時不依賴 `statsmodels`。主要結果：

| 變數 | standardized coefficient | SE | p-value |
| --- | ---: | ---: | ---: |
| `lambda` | 0.039 | 0.005 | 約 2.89e-15 |
| `lambda x rain` | -0.006 | 未列入摘要 | 0.63 |

解讀：

1. 每車需求越高，corrected synchronization `r_excess` 越高。
2. 效果大小不大，但統計上穩定。
3. 雨天沒有明顯放大 demand effect。

### Transition / critical load 初步結果

Phase 2 也做了 piecewise model，找到表面最佳 change point：

```text
lambda_c = 135.74 boardings / bus
```

但 Phase 2 已保留謹慎語氣：這只是 observational candidate critical load，不是因果結論。Phase 3 後來用更嚴格的 gate 把它降級為 smooth crossover，而不是真正 critical transition。

### Weather modifier

Phase 2 比較 rain vs dry：

| 指標 | rain minus dry |
| --- | ---: |
| mean `r_excess` difference | -0.008 |
| 95% CI | -0.038 到 0.024 |

結論：沒有證據顯示雨天顯著改變 corrected synchronization。

### 初步 corridor coupling

Phase 2 用 shared route overlap 比較 shared pairs 與 non-shared pairs：

| 類別 | 數值 |
| --- | ---: |
| Shared pairs | 176 |
| Non-shared pairs | 14 |
| Mean corr shared | 0.039 |
| Mean corr non-shared | 0.049 |
| Label permutation p | 0.771 |

這個設計不平衡，non-shared pairs 太少，所以結論是 inconclusive。這不是「沒有 corridor coupling」的證據，而是這個 early test 本身設計不足。Phase 3 因此改成 within-pair shared trunk vs off-trunk 設計。

### Phase 2 主要輸出

| 檔案 | 內容 |
| --- | --- |
| `outputs/phase2/line_selection.csv` | 候選與最終選入路線 |
| `outputs/phase2/map_matching_quality_phase2.csv` | 最終選入路線 map-matching 品質 |
| `outputs/phase2/filtering_and_exclusion_log.json` | bus-hour 排除紀錄 |
| `outputs/phase2/bus_hour_phase_candidates.csv` | active bus-hour 相位候選資料 |
| `outputs/phase2/hourly_order_raw.csv` | hourly raw order |
| `outputs/phase2/finite_size_null_by_N.csv` | same-N null simulation |
| `outputs/phase2/phase2_hourly_observables.csv` | Phase 2 主表，11,641 rows |
| `outputs/phase2/model_frame.csv` | 模型用資料框 |
| `outputs/phase2/mixed_model_r_excess_coefficients.csv` | `r_excess` 模型係數 |
| `outputs/phase2/mixed_model_z_coefficients.csv` | `z` 模型係數 |
| `outputs/phase2/transition_model_comparison.csv` | transition/piecewise model 比較 |
| `outputs/phase2/weather_matched_dry_rain_pairs.csv` | 天氣 matched pairs |
| `outputs/phase2/corridor_pair_overlap_and_correlation.csv` | corridor pair correlation |

## Phase 3 做了什麼

### 目標

Phase 3 的目標是把 Phase 2 的「candidate critical load」正式檢驗：到底是 sharp critical transition，還只是平滑 crossover？同時 Phase 3 加入三個物理機制層面的分析：

1. dwell variability 是否中介 demand -> bunching。
2. headway irregularity 是否沿路線下游放大。
3. shared corridor 訊號在 within-pair 設計下是否存在。

### Phase 3 主表

Phase 3 主表：

```text
outputs/phase3/phase3_observables.csv
```

它延續 Phase 2 的 11,641 rows，擴充到 50 columns。

### Criticality gate

我在 Phase 3 設定預先承諾的判準：至少 3 個 signature 中通過 2 個，才可稱為 genuine transition。

三個 signature 是：

| Signature | 要檢查什麼 |
| --- | --- |
| Susceptibility peak within N strata | 在固定 `N` 分層中，波動是否在某個 `lambda` 附近明顯尖峰 |
| Bimodality near candidate lambda | 在候選 `lambda_c` 附近是否有兩態共存 |
| N-stability of apparent threshold | apparent threshold 是否不只是 `N` 改變造成 |

結果：

| Signature | 是否通過 |
| --- | --- |
| Susceptibility | False |
| Bimodality | False |
| N-stability | True |
| 通過數 | 1 / 3 |

因此 Phase 3 的 verdict 是：

```text
consistent with a smooth crossover / no criticality
```

也就是說，Phase 2 的 `lambda_c = 135.74 boardings/bus` 不能被解讀為真正臨界點，只能視為 observational crossover scale。

### Susceptibility 細節

各 `N` strata 的 peak/edge ratio 都沒有達到強尖峰標準：

| N stratum | peak/edge ratio | lambda_peak |
| --- | ---: | ---: |
| N13_plus | 0.891 | 12.962 |
| N09_12 | 1.139 | 55.894 |
| N04_05 | 1.072 | 11.200 |
| N02_03 | 1.059 | 4.000 |
| N06_08 | 1.127 | 20.286 |

這表示資料中沒有穩定、尖銳、跨 N 可重現的 susceptibility peak。

### Mixture / bimodality 細節

候選 `lambda_c` 附近 two-component mixture 的 BIC improvement 是 17.996，但 low/high regimes 也有更大的 improvement。這代表 mixture evidence 不是特定出現在 candidate threshold 附近，因此不能當作臨界點附近兩態共存的證據。

### Dwell 與 non-stop delay

Phase 3 從 GPS 低速段估計 dwell：

| 定義 | 規則 |
| --- | --- |
| stop dwell candidate | GPS speed <= 1.5 m/s 且距離 official stop <= 50 m |
| dwell summary inclusion | 排除 <15 s 或 >300 s events |
| long terminal-like dwell | 作為 layover 排除 |
| non-stop delay | 低速但不在 stop 附近 |

偵測結果：

| 類別 | 數量 |
| --- | ---: |
| Dwell events | 1,875,802 |
| Non-stop delay events | 1,430,536 |
| Low-speed intervals near stops | 4,755,025 |
| Low-speed intervals away from stops | 5,339,664 |

### Mediation 結果

Phase 3 檢查：

```text
lambda -> dwell CV -> r_excess
```

模型 rows：10,082。

| 指標 | 數值 |
| --- | ---: |
| Indirect effect | 0.0001 |
| 95% CI | -0.0002 到 0.0004 |
| Proportion mediated | 0.002 |

結論：GPS-derived dwell variability 沒有可偵測到的 mediation effect。這不代表 dwell 完全不重要，而是以目前 noisy GPS dwell proxy，無法支持「dwell variability 是 demand 導致 bunching 的主要中介機制」。

### Spatial amplification

Phase 3 把 cross sections 加密：

```text
5%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 95%
```

斜率估計使用 interior 10% 到 90%。目標是看 headway CV 是否沿路線下游增加。

結果：

| 指標 | 數值 |
| --- | ---: |
| Slope rows | 6,669 |
| Mean headway-CV slope | 0.1198 |
| Median headway-CV slope | 0.1581 |
| One-sample t-test p | 1.93e-58 |
| Lambda coefficient for slope | -0.0391 |

結論：

1. 下游 amplification 明確存在。
2. 但 amplification 沒有隨 demand 增強；`lambda` 係數反而為負。
3. 這支持「服務不規則會沿線累積」的空間不穩定圖像，但不支持「需求越高就越接近臨界 amplification」。

### Within-pair corridor coupling

Phase 2 的 shared vs non-shared 設計不夠好，所以 Phase 3 改成同一對 line pair 內比較：

```text
shared trunk correlation vs off-trunk correlation
```

結果：

| 指標 | 數值 |
| --- | ---: |
| Pairs tested | 176 |
| Shared corr | 0.051 |
| Off-trunk corr | -0.005 |
| Shared minus off | 0.055 |
| Sign-flip p | 0.000 |

這表示 shared trunk 上確實有更高同步訊號。但這個訊號仍可能來自共同道路狀態，例如同一 shared segment 上的速度變慢、塞車、號誌或道路干擾。這正是 Phase 4 要處理的問題。

### Phase 3 主要輸出

| 檔案 | 內容 |
| --- | --- |
| `outputs/phase3/phase3_observables.csv` | Phase 3 主表，11,641 rows，50 columns |
| `outputs/phase3/criticality_summary.json` | criticality gate 結果 |
| `outputs/phase3/criticality_susceptibility_by_N_lambda.csv` | susceptibility by N/lambda |
| `outputs/phase3/criticality_apparent_lambda_by_N.csv` | apparent lambda peaks by N |
| `outputs/phase3/criticality_mixture_bimodality.csv` | mixture/bimodality 檢查 |
| `outputs/phase3/dwell_events.csv` | GPS-derived dwell events |
| `outputs/phase3/non_stop_delay_events.csv` | non-stop low-speed delay events |
| `outputs/phase3/dwell_hourly.csv` | hourly dwell features |
| `outputs/phase3/mediation_bootstrap.csv` | mediation bootstrap |
| `outputs/phase3/mediation_summary.json` | mediation 結論 |
| `outputs/phase3/headway_cv_by_section_hour.csv` | section-hour headway CV |
| `outputs/phase3/spatial_amplification_hourly.csv` | spatial amplification slopes |
| `outputs/phase3/segment_local_order.csv` | segment-local order |
| `outputs/phase3/corridor_within_pair_coupling.csv` | within-pair corridor coupling |

Phase 3 圖：

| 圖 | 檔案 |
| --- | --- |
| Criticality gate | `outputs/phase3/p3_1_criticality_gate.png` |
| Mediation dwell | `outputs/phase3/p3_2_mediation_dwell.png` |
| Spatial amplification | `outputs/phase3/p3_3_spatial_amplification.png` |
| Corridor coupling within pair | `outputs/phase3/p3_4_corridor_coupling_within_pair.png` |
| Updated demand coupling | `outputs/phase3/p3_5_updated_demand_coupling.png` |

## Phase 4 做了什麼

### 目標

Phase 4 的目標是做最後的 common-shock control 與 physical synthesis。Phase 3 已經看到 shared trunk 訊號，但還不能判斷這是路線之間真的互相耦合，還是同一段路共同受到塞車、速度下降、號誌或道路狀態影響。

Phase 4 因此做兩件事：

1. 對 corridor coupling 做 speed-control 與 lead-lag/transfer-entropy gate。
2. 建立 spatial amplification driver model，判斷下游 CV 成長主要由哪些因素驅動。

### Phase 4 主表

Phase 4 主表：

```text
outputs/phase4/phase4_observables.csv
```

它延續 Phase 2/3 的 11,641 rows，擴充到 63 columns。

### Pre-committed coupling decision rule

Phase 4 設定兩道 gate：

| Gate | 通過條件 |
| --- | --- |
| Speed survival | speed-controlled shared-minus-off gap 為正、bootstrap 95% CI 排除 0、且保留至少 50% raw gap |
| Lead-lag | 至少 25% pairs 有顯著 non-zero-lag peak，或 signed net transfer-entropy asymmetry 超過 pair-shuffle null |

判定規則：

| 結果 | 解讀 |
| --- | --- |
| 兩者都通過 | coupling survives common-shock controls |
| 兩者都不通過 | largely common-shock |
| 只通過一個 | mixed |

### Speed-control 方法

Phase 4 建立 shared-segment hourly speed proxy：

1. 取 Phase 2 active bus-hour GPS movement summaries。
2. 找出 midpoint 位於 pair shared segment 內的 bus-hours。
3. 聚合出 shared trunk 的 hourly mean speed。
4. 將 shared trunk line series 對 hourly shared-segment mean speed residualize。
5. 比較 residualized shared-minus-off correlation gap。

這是一個 approximation，因為它不是真正道路感測器速度，而是 GPS-derived bus movement speed proxy。但它能測試 shared corridor 訊號是否只是共同慢速道路狀態。

### Coupling gate 結果

| 指標 | 數值 |
| --- | ---: |
| Pairs tested | 156 |
| Raw shared-minus-off gap | 0.055 |
| Raw 95% CI | 0.028 到 0.085 |
| Speed-controlled gap | 0.018 |
| Speed-controlled 95% CI | -0.015 到 0.054 |
| Retention | 0.33 |
| Speed survival pass | False |

speed control 後，gap 只保留 33%，而且 CI 包含 0，所以沒有通過 speed survival。

Lead-lag / transfer entropy：

| 指標 | 數值 |
| --- | ---: |
| Significant non-zero-lag fraction | 0.058 |
| Typical significant lag | -1.0 h |
| Mean absolute TE asymmetry | 0.0466 |
| Signed net TE asymmetry | 0.0011 |
| Signed permutation p | 0.874 |
| Lead-lag pass | False |

兩個 gate 都沒有通過，所以 Phase 4 verdict 是：

```text
largely common-shock
```

這是 Phase 4 最重要的結論：shared corridor 有 raw synchronization 訊號，但大部分可以用共同道路速度狀態解釋。剩下的小殘差不足以支持強的 propagating inter-line coupling claim。

### Amplification driver model

Phase 4 也檢查 spatial amplification 的 driver。模型資料：

| 指標 | 數值 |
| --- | ---: |
| Model rows | 5,544 |
| Line-direction groups | 38 |

標準化 random-intercept 係數：

| term | estimate | 95% CI | p-value | 解讀 |
| --- | ---: | --- | ---: | --- |
| `initial_headway_cv_std` | -0.283 | -0.296 到 -0.270 | 0.000 | 最大 driver，負向 |
| `N_std` | 0.081 | 0.066 到 0.096 | 0.000 | 車輛數越高，amplification slope 較高 |
| `lambda_boardings_per_bus_std` | 0.015 | 0.002 到 0.027 | 0.021 | demand 有小的正向效果 |
| `route_length_km_std` | 0.040 | -0.022 到 0.103 | 0.205 | 不顯著 |
| `sinuosity_std` | -0.070 | -0.154 到 0.015 | 0.105 | 不顯著 |
| `stop_density_per_km_std` | -0.005 | -0.078 到 0.068 | 0.887 | 不顯著 |

最大 driver 是 `initial_headway_cv_std`，而且係數是負的。這代表一開始就已經很不規則的 hours，後面可再增加的 CV 較少。這比較像 saturation 或 ceiling effect，而不是 demand 觸發的臨界增長。

### Phase 4 圖與輸出

Phase 4 設定 publication-grade matplotlib rcParams，並輸出 300 dpi PNG 與 PDF，也為每張圖輸出 caption txt。

主要輸出：

| 檔案 | 內容 |
| --- | --- |
| `outputs/phase4/phase4_observables.csv` | Phase 4 主表，11,641 rows，63 columns |
| `outputs/phase4/segment_speed_bus_hour_observations.csv` | shared segment speed 的 bus-hour observations |
| `outputs/phase4/segment_speed_pair_hour.csv` | pair-hour speed proxy |
| `outputs/phase4/corridor_coupling_speed_controlled.csv` | speed-controlled corridor coupling |
| `outputs/phase4/corridor_coupling_gate_summary.json` | coupling gate verdict |
| `outputs/phase4/route_topology_features.csv` | route topology features |
| `outputs/phase4/amplification_driver_model_frame.csv` | driver model frame |
| `outputs/phase4/amplification_driver_coefficients.csv` | driver coefficients |
| `outputs/phase4/amplification_driver_summary.json` | driver model summary |
| `outputs/phase4/phase4_counts_and_diagnostics.json` | counts and diagnostics |

Phase 4 figures：

| 圖 | PNG | PDF | Caption |
| --- | --- | --- | --- |
| Corridor speed gate | `outputs/phase4/figures/p4_1_corridor_speed_gate.png` | `outputs/phase4/figures/p4_1_corridor_speed_gate.pdf` | `outputs/phase4/figures/p4_1_corridor_speed_gate_caption.txt` |
| Lead-lag / TE | `outputs/phase4/figures/p4_2_lead_lag_transfer_entropy.png` | `outputs/phase4/figures/p4_2_lead_lag_transfer_entropy.pdf` | `outputs/phase4/figures/p4_2_lead_lag_transfer_entropy_caption.txt` |
| Amplification drivers | `outputs/phase4/figures/p4_3_amplification_drivers.png` | `outputs/phase4/figures/p4_3_amplification_drivers.pdf` | `outputs/phase4/figures/p4_3_amplification_drivers_caption.txt` |
| Headway CV arc / initial CV | `outputs/phase4/figures/p4_4_headway_cv_arc_initial_cv.png` | `outputs/phase4/figures/p4_4_headway_cv_arc_initial_cv.pdf` | `outputs/phase4/figures/p4_4_headway_cv_arc_initial_cv_caption.txt` |
| Synthesis | `outputs/phase4/figures/p4_5_synthesis.png` | `outputs/phase4/figures/p4_5_synthesis.pdf` | `outputs/phase4/figures/p4_5_synthesis_caption.txt` |

## 四個 Phase 的邏輯演進

### 從 raw synchronization 到 corrected synchronization

Phase 1 先建立 raw `r`，但發現 weekend raw `r` 偏高。Phase 2 顯示這不是物理上 weekend 更 bunching，而是 `N` 小時 raw `r` 自然偏高。因此後續主分析都改用 `r_excess` 或 `z`。

這是整個專案最重要的方法轉折之一：

```text
不要直接比較不同 N 的 raw r。
必須用 same-N random phase null correction。
```

### 從 candidate critical load 到 smooth crossover

Phase 2 的 piecewise model 找到 candidate `lambda_c = 135.74 boardings/bus`。但 Phase 3 用 susceptibility、bimodality、N-stability 三個 gate 檢查後，只有 N-stability 通過。因此不能宣稱 critical transition。

最後結論改成：

```text
demand-modulated bunching exists, but it behaves like a smooth crossover.
```

### 從 shared corridor correlation 到 common-shock verdict

Phase 3 within-pair 設計顯示 shared trunk correlation 高於 off-trunk，shared-minus-off gap 是 0.055。但 Phase 4 加入 shared-segment speed control 後，gap 降到 0.018 且 CI 包含 0，再加上 lead-lag/TE 不支持方向性傳播。

所以最終不把這解讀成強 inter-line coupling，而是：

```text
shared corridor synchronization is largely common-shock.
```

### 從 demand mechanism 到 spatial instability

Phase 2 顯示 demand per bus 對 corrected bunching 有小的正向關係。Phase 3 顯示 downstream headway-CV amplification 非常穩定存在。Phase 4 顯示 amplification 的最大 driver 不是 demand，而是 initial headway CV 的 negative saturation/ceiling pattern。

因此最後物理圖像不是「需求單獨造成臨界同步」，而是：

```text
bus service irregularity is spatially amplified along routes,
while demand is one modest modulator among several operational factors.
```

## 最終可支持與不可支持的主張

### Supported

1. Same-N finite-size correction 是必要的。不同 `N` 的 raw Kuramoto `r` 不能直接比較。
2. 修正後，weekday/weekend 的結論與 raw `r` 相反：weekend corrected synchronization 較低。
3. `lambda = boardings / N` 與 corrected bunching 有小但穩定的正相關。
4. 下游 headway-CV amplification 明確存在。
5. Niteroi 資料較符合 demand-modulated smooth crossover，而不是 sharp critical transition。
6. Shared corridor raw synchronization 訊號存在，但 Phase 4 顯示主要是 common-shock。

### Suggestive

1. Shared trunk 仍有小的 residual speed-controlled component，但不夠穩健，不能稱為主要傳播耦合。
2. Demand 可能參與服務不穩定，但不是唯一或主導 driver。
3. Route topology features 目前沒有 strong driver evidence，但仍可能需要更好的設計或更長觀測期。

### Null / Not supported

1. 不支持「passenger demand alone produces a critical synchronization transition」。
2. 不支持把 Phase 2 的 `lambda_c` 當成真正臨界點。
3. 不支持 GPS-derived dwell variability 是 demand -> corrected bunching 的主要中介。
4. 不支持 weather/rain 明顯改變 demand-bunching 關係。
5. 不支持在 Phase 4 資料中有強烈、方向性的 inter-line propagation coupling。

## 主要限制

1. Ticketing 和 mobility 車號沒有 overlap，所以不能估計車輛層級 passenger load。
2. `lambda = boardings / N` 是 line-hour proxy，不是每台車真實載客。
3. GPS-derived dwell 是 proxy，沒有 door-open 或 stop-level APC ground truth。
4. Shared segment speed proxy 來自 bus GPS movement，不是獨立道路速度感測器。
5. Phase 4 common-shock control 能削弱 inter-line coupling claim，但不能證明所有殘差完全沒有物理互動。
6. 這些結果是 observational，不是 randomized intervention 或 causal experiment。

## 最後的研究敘事

我做的整體工作可以整理成以下研究敘事：

1. 先從 GPS 和 ticketing 原始資料建立可重現 pipeline。
2. 把車輛 GPS map match 到路線 geometry，將沿線位置轉成 Kuramoto phase。
3. 建立 headway、bunching、demand、order parameter 等 observables。
4. 發現 raw synchronization 有 finite-size bias。
5. 用 same-N random phase null 修正 `r`，證明 Phase 1 weekend paradox 是 artifact。
6. 擴大到 20 條高需求且 map-matching 合格的路線與 19 個可用日期。
7. 證明 demand per bus 與 corrected bunching 有小的正向 association。
8. 將 candidate critical load 用更嚴格 gate 檢查，最後否定 sharp criticality。
9. 從 GPS 低速事件建立 dwell/non-stop delay proxy，但沒有找到 dwell mediation。
10. 建立沿路線 cross-section 的 headway CV，證明 irregularity 會 downstream amplify。
11. 先用 within-pair 設計找到 shared trunk synchronization 訊號。
12. 再用 Phase 4 speed control 與 lead-lag/TE gate 顯示該訊號 largely common-shock。
13. 最後形成 publication-ready synthesis：這是一個 spatial/networked service instability 問題，不是一個需求單獨觸發的臨界同步轉變。

## 檢查用 row counts

目前主要報告與主表的行數如下，CSV 行數包含 header：

| 檔案 | `wc -l` | 實際資料 rows |
| --- | ---: | ---: |
| `phase1_report.md` | 141 | 不適用 |
| `phase2_report.md` | 194 | 不適用 |
| `phase3_report.md` | 99 | 不適用 |
| `phase4_report.md` | 73 | 不適用 |
| `outputs/phase1/phase1_line_timebin_observables.csv` | 5,803 | 5,802 |
| `outputs/phase2/phase2_hourly_observables.csv` | 11,642 | 11,641 |
| `outputs/phase3/phase3_observables.csv` | 11,642 | 11,641 |
| `outputs/phase4/phase4_observables.csv` | 11,642 | 11,641 |

