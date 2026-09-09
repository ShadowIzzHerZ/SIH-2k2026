# Smart India Hackathon (SIH) 2026 — Strategy & Problem Selection Dossier

*Last updated: August 31, 2026*
*Status: **LOCKED IN** — building for **SIH26168** — AI/ML-based Intelligent Dead Reckoning for Seamless Navigation (ISRO)*

---

## 1. Executive Summary

SIH 2026 has **229 official problem statements** across 17 themes (175 Software, 54 Hardware) — earlier drafts of this doc said 226/18; corrected against the live scraped catalogue on 2026-08-29.

This doc previously centered on **SIH26070 (cyclone prediction)**. That pick has been **dropped** after two problems surfaced:
1. Its `dataset_link` field is empty on the official listing — MOSDAC/IMD access is real but not organizer-guaranteed, unlike the pick below.
2. Its stated theme was inconsistently reported across sources (seen as both "Smart Education" and "Disaster Management") — a red flag for how cleanly it's actually classified. Verify any PS's theme yourself on sih.gov.in before building a pitch around it.

The current primary pick, **SIH26168**, was re-evaluated against the actual SIH judging guidance — *"a well-scoped, deeply understood problem consistently beats an ambitious one your team can't fully execute"* — and it scores best on that axis: narrowest scope, a real benchmark dataset, and a textbook (not research-grade) ML technique.

---

## 2. The exact problem statement — SIH26168

Verified against the official SIH 2026 catalogue (scraped 2026-08-29).

| Field | Value |
|---|---|
| **PS Number** | SIH26168 |
| **Title** | AI-ML based Intelligent Dead Reckoning system for seamless navigation |
| **Organization** | Indian Space Research Organisation (ISRO), Department of Space |
| **Category** | Software |
| **Theme** | Smart Vehicles |
| **Dataset** | [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) — Inertial & Odometry benchmark dataset for ground vehicle positioning (official, organizer-provided) |
| **Idea submission deadline** | 20 September 2026 |

**✅ Verified directly against the live `sih.gov.in` table source (not a mirror)** — row 168 confirms `S.No: 168 | Organization: Indian Space Research Organisation(ISRO) | Title: AI-ML based Intelligent Dead Reckoning system for seamless navigation | PS Number: SIH26168`. The [IO-VNBD dataset repo](https://github.com/onyekpeu/IO-VNBD) it points to was also confirmed real, with a README matching the claimed content (40hrs/1,300km vehicle data + 58hrs/4,400km smartphone data across UK/Nigeria/France).

### Problem, in plain terms
Phone GPS drops out in tunnels, underground parking, dense urban canyons, and forests. Every vehicle without a factory-fitted INS (i.e. almost all two-wheelers, older cars, trucks — the vast majority of vehicles on Indian roads) then relies on the driver's phone, which has no reliable way to keep tracking position once GPS is gone. The ask: use the phone's own accelerometer/gyroscope (IMU) to keep tracking position through the blackout, using AI/ML — not just raw physics — because raw IMU integration drifts wildly within seconds.

### What the deliverable actually has to be
This is more than a training-notebook problem — read carefully, because it changes team-skill requirements:
- A **working mobile application** (not just a model) that runs live IMU sensor fusion **on-device**, in real time.
- An **edge-deployable software engine** version that also works with non-phone IMU sensors (not just the mobile app).
- Concrete, numeric performance benchmarks you are graded against:
  - Positional drift **< 10%** of distance traveled during GNSS blackout (e.g. <5m drift over 50m in <1 min, or <100m drift over 1km at 60kmph)
  - Position update rate of **10Hz on the phone**, ~200Hz on the FOG-IMU edge engine
- Named technical components expected: in-vehicle phone alignment/calibration, an AI speed/vibration filter, map-matching against OpenStreetMap with non-holonomic constraints, a GNSS+INS fusion engine, and instant seamless mode-switching.
- Training happens beforehand (cloud/desktop, on IO-VNBD or your own collected data); inference must run live on a smartphone at the finale.

**Team-skill implication:** you need someone who can ship a real Android/iOS app reading live IMU sensors, not just a Python notebook. If your team is purely ML/backend with no mobile dev experience, budget real time for that gap or bring in a teammate who has shipped a mobile app before.

---

## 3. Why this is a genuinely strong pick — and where I'm not just agreeing with you

**Real strengths, verified, not assumed:**
- Organizer-provided, real, working dataset (IO-VNBD) — most PS on the full 229-list don't have one at all (only 44/229 do).
- Scope is narrow and single-purpose: one sensor-fusion problem, not a multi-stage pipeline bolted onto other features. This matters — SIH judging rewards "well-scoped and fully executed" over "ambitious and half-working."
- Clear, decades-old technique family (Kalman filtering / IMU sensor fusion) with an ML layer on top — not asking you to invent new science.
- Concrete numeric success criteria are given upfront (<10% drift), so you know exactly what "done" looks like — most PS don't give you that.

**Where I'd push back before you commit:**
- The on-device, real-time, 10Hz mobile-app requirement is a real engineering lift most teams underweight — this is not "train a model and show a plot," it's "ship a live sensor-fusion Android app that doesn't lag." If nobody on the team has done Android sensor work, this is the single biggest risk to the plan, bigger than the ML itself.
- "Low competition" (my earlier claim, and your original doc's) is **unverified** — idea-submission counts are 0/500 across the entire catalogue right now because the window is still open. Recheck actual counts closer to 20 Sept; don't build team morale on a competition assumption I can't currently back with data.
- Meeting the <10% drift benchmark for real, on a live smartphone, in front of judges, is harder than it sounds — cheap MEMS IMUs are genuinely noisy. Budget serious testing time, not just build time.

---

## 4. Technical stack

### Mobile / On-device
- **Platform:** Android (Kotlin) or Flutter — pick based on team's existing mobile experience, don't learn a new mobile framework mid-hackathon
- **Sensor access:** Android SensorManager (accelerometer, gyroscope, magnetometer) at max sampling rate
- **On-device inference:** TensorFlow Lite / ONNX Runtime Mobile (export trained model to a lightweight runtime)

### ML / Sensor Fusion
- **Training framework:** PyTorch or TensorFlow, trained offline on IO-VNBD
- **Core techniques:** Unscented Kalman Filter (UKF) or learned bias-correction network for IMU drift, Hidden Markov / weighted map-matching for road-snapping
- **Map matching:** OpenStreetMap extract for the demo route, non-holonomic motion constraints (car can't move sideways/vertically)

### Edge engine (non-phone IMU path)
- **Language:** Python or C++ service consuming raw IMU streams, same fusion model exported for a non-mobile runtime

### Demo / Visualization
- Live map view (Mapbox/MapLibre or Google Maps SDK on-device) showing the vehicle icon tracking smoothly through a simulated GNSS blackout (e.g. walk/drive through an underground parking structure or tunnel during the demo)

---

## 5. 36-Hour Execution Roadmap

```mermaid
gantt
    title 36-Hour SIH Implementation Schedule — SIH26168
    dateFormat  HH
    axisFormat  Hour %H

    section Data & Model
    IO-VNBD preprocessing + own IMU data collection :00, 05h
    Sensor fusion model training (UKF / learned bias):05, 08h
    Map-matching + non-holonomic constraint layer   :13, 05h

    section Mobile App
    Android sensor pipeline + calibration engine    :00, 08h
    On-device model export (TFLite/ONNX) + inference :08, 06h
    Live map UI + seamless GNSS/INS mode-switch      :14, 06h

    section Integration & Polish
    End-to-end blackout simulation testing (real route) :20, 08h
    Drift-benchmark validation against <10% target      :28, 04h
    Pitch deck, 3-min script, demo rehearsal             :32, 04h
```

---

## 6. Backup shortlist — if SIH26168 falls through

Ranked by actual likelihood of a team fully executing a working demo (not just "coolest tech"), based on full problem-description review, not just titles:

| # | PS | Title | Org | Note |
|---|---|---|---|---|
| 2 | SIH26162 | AI Detection/Classification of Industrial Fires (NASA FIRMS + OSM) | NTRO | Narrow, live public API, real-time demo |
| 3 | SIH26056 | Real-time Airfare Price Index via web scraping, for CPI | MoSPI | Data engineering, not deep learning — low technical risk |
| 4 | SIH26102 | AI fraud/anomaly detection in MPLAD Scheme spending | MoSPI | Standard anomaly detection on real gov data |
| 5 | SIH26101 | AI learning platform for iGOT Karmayogi | MoSPI | Buildable on an LLM pipeline, low research risk |
| 6 | SIH26142 | Deep-learning super-resolution mapping from satellite imagery | NTRO | Well-defined single task, existing architectures |
| — | SIH26143 | Oil spill detection + AIS vessel attribution | NTRO | High wow, but 4 chained hard sub-problems — only if scoped down |
| — | SIH26167 | SatQuery AI — vision-language search over satellite imagery | ISRO | Genuinely research-grade multimodal system — only for a strong ML team |
| — | SIH26175 | DepthWizard — single-view height estimation + 3D flythrough | ISRO | Org's own text admits the core sub-problem is unsolved research |
| — | SIH26153 | AI-based Network Attack Forecasting | NTRO | Asks for World Models + GNN + explainability at once — ambitious |

Not carried forward: SIH26070 (cyclone) — no organizer dataset, theme-tag inconsistency across sources.

---

## 7. What you need — full requirements list

### 7.1 Team roles (6-person team, ISRO's usual SIH cap)
- [ ] **Mobile/Android engineer** — non-negotiable. Someone who has shipped an app reading live device sensors before. This is the #1 gap most ML-focused teams have; fill it first.
- [ ] **ML engineer** — sensor fusion / time-series background helps more than generic deep learning experience (Kalman filters, signal processing > CNNs here).
- [ ] **Backend/edge engineer** — builds the non-phone "edge-deployable engine" variant the PS explicitly requires as a second deliverable.
- [ ] **Geospatial/maps person** — OpenStreetMap extraction, map-matching, live map UI.
- [ ] **Pitch/presentation owner** — someone who isn't heads-down coding in the last 4 hours, dedicated to the demo script and slides.
- [ ] 1 floater / tester — runs the live blackout-simulation tests repeatedly while others keep building.

### 7.2 Hardware
- [ ] 2+ Android phones (mid-range is fine, but IMU quality varies by device — test on the *actual* phone you'll demo with, not a borrowed one on the day)
- [ ] Phone dashboard/holder mount (the PS explicitly covers both dashboard-mounted and hand-held-in-holder alignment)
- [ ] A vehicle (car or two-wheeler) for real IMU data collection and later live blackout testing — or at minimum access to a tunnel/basement parking structure to simulate GNSS blackout
- [ ] Laptops with a GPU (local or cloud) for model training — a UKF/small fusion network doesn't need much, but leave headroom
- [ ] Reliable mobile data/hotspot for the venue (for OSM tile fetching, unless you pre-download the map extract — you should)

### 7.3 Software & dev environment (install/set up *before* hackathon day)
- [ ] Android Studio + Kotlin (or Flutter SDK, if that's the team's stronger stack) — pick one now, don't decide on the day
- [ ] Python 3.11+, PyTorch or TensorFlow, NumPy/SciPy, `filterpy` or similar (UKF implementation)
- [ ] TensorFlow Lite converter / ONNX Runtime Mobile — the exact path from trained model → on-device inference
- [ ] OSM tooling: `osmnx` or raw OSM extract for the demo route/venue area, offline map renderer (Mapbox GL / MapLibre / Google Maps SDK — confirm API key limits for offline/demo use)
- [ ] Git repo + CI-free local build pipeline (don't rely on hackathon wifi for CI)

### 7.4 Accounts / access to arrange in advance
- [ ] Map SDK API key (Mapbox or Google Maps) with usage confirmed under whatever free/demo tier you'll use live
- [ ] GitHub org/repo set up for the team
- [ ] SIH portal team registration + idea submission access (official deadline **20 Sept 2026** — don't leave this to the last day)
- [ ] **College internal idea shortlist — 10 Sept 2026** — your college/SPOC's own review cutoff, ahead of and separate from the official SIH portal deadline above. This is the one that actually constrains your working timeline; treat it as the real deadline day-to-day.

### 7.5 Datasets & assets to pre-download
- [ ] [IO-VNBD dataset](https://github.com/onyekpeu/IO-VNBD) — the official organizer-provided dataset, pull it now, don't wait for venue wifi
- [ ] Your own collected IMU dataset — record real accelerometer/gyro/GNSS logs from an actual drive (car + phone mounted), several sessions, varied speeds/roads — real recorded data beats synthetic here
- [ ] OSM extract for your planned demo route, downloaded offline
- [ ] A pre-scouted GNSS-blackout location (underground parking, tunnel) to rehearse the live demo in, ideally the same one you'll use at the venue if possible

### 7.6 Pre-hackathon prep (do before the 36h clock starts — per the PS itself)
- [ ] Train the initial fusion model on IO-VNBD *before* arriving — the PS explicitly expects teams to "bring trained models" and only fine-tune/extend during the event
- [ ] Get a bare-bones version of the mobile sensor-read + inference pipeline working end-to-end early, even before the model is good — the *plumbing* (sensor → model → map UI) is what's most likely to eat unplanned time, not the ML accuracy
- [ ] Run at least one real blackout test (drive/walk into a tunnel or basement) and measure actual drift against the <10% target — do this days before, not hours before

### 7.7 Still open / unverified — check yourself
- [ ] Confirm SIH26168's theme/category directly on sih.gov.in (mirrors have been inconsistent for other PS — verify this one too)
- [ ] Recheck idea-submission counts closer to 20 Sept 2026 for a real (not guessed) competition-level read

---

## 8. Price list — what everything actually costs (₹ INR)

Everything the ML/software stack needs is free/open-source. The real costs are hardware and a couple of optional buffers. Ranges assume Indian retail pricing, checked against general market rates, not live-quoted — re-check before actually buying.

| # | Item | Est. Cost (₹) | Required? | Notes |
|---|---|---:|---|---|
| 1 | Android phone (dedicated test/demo device) | ₹0 – 12,000 | Only if team doesn't already own a spare | ₹0 if you use an existing team member's phone; budget ₹8k–12k for a second-hand mid-range phone as a dedicated test unit so you're not risking someone's daily phone |
| 2 | Phone dashboard/holder mount | ₹200 – 500 | **Yes** | Cheap, buy 2 (one for testing, one spare for demo day) |
| 3 | Vehicle fuel / auto rides for real data-collection drives | ₹500 – 1,500 | **Yes** | Several short drives to record real IMU+GNSS logs |
| 4 | Laptop with GPU | ₹0 | Usually already owned | If none has a GPU, Google Colab free tier covers small model training — see below |
| 5 | Cloud GPU (Google Colab Pro, optional) | ₹0 – 850/month | Optional | Free tier (₹0) is enough for a small UKF/fusion network; only pay if training is slow |
| 6 | Mobile data / hotspot pack for venue | ₹300 – 500 | **Yes** | For OSM tile fetch fallback and general connectivity at the hackathon venue |
| 7 | Map SDK (Mapbox or Google Maps API) | ₹0 | **Yes** | Free tier: 50,000 map loads/month — nowhere near what a demo needs. Note: Mapbox asks for a card on file even for the free tier. |
| 8 | Android Studio / Flutter SDK, PyTorch, TensorFlow Lite, OSM tooling | ₹0 | **Yes** | All free, open-source |
| 9 | GitHub repo | ₹0 | **Yes** | Free tier covers a student team easily |
| 10 | IO-VNBD dataset | ₹0 | **Yes** | Free public download |
| 11 | SIH portal registration | ₹0 (unverified) | **Yes** | No participation fee found in current SIH 2026 materials — **confirm with your college SPOC**, don't take this as fully confirmed |
| 12 | Travel/lodging to nodal centre or grand finale | ₹0 – 5,000+ | Situational | Varies hugely by distance; SIH sometimes covers finalist travel/lodging but this isn't guaranteed at every stage — **confirm with SPOC**, don't assume it's covered |
| 13 | Team merch (T-shirts, printed banner for pitch) | ₹0 – 3,000 | Optional | Doesn't affect judging; purely team morale/photos |

### Budget scenarios
- **Bare minimum** (team already owns phones/laptops, no travel, no merch): **≈ ₹1,000 – 2,500**
- **Recommended** (dedicated test phone bought second-hand, mounts, fuel, data pack): **≈ ₹9,500 – 15,000**
- **Comfortable** (adds Colab Pro buffer + merch + some travel budget): **≈ ₹15,000 – 22,000+**

The bottleneck was never money on this problem statement — it's the mobile engineering skill gap flagged in §7.1. Don't over-invest in hardware before that's solved.
