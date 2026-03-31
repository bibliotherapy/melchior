# GMFCS Level Classification AI 모델 — Comprehensive Report

**작성일**: 2026-03-20 (updated)
**목적**: 6세 이하 소아 뇌성마비 환아 비디오 기반 GMFCS Level Classification, Inference 정확도 80% 이상 달성

---

## 1. Executive Summary

### 프로젝트 개요

삼성서울병원 소아 물리치료실에서 촬영한 24명의 6세 이하 뇌성마비(CP) 환아의 다중 시점 비디오를 활용하여, AI 모델로 GMFCS Level(1~5)을 자동 분류하는 시스템을 구축한다. 현재 서버(V100 GPU × 2)에서 학습 중인 모델은 Epoch 6/20 기준 Overall Accuracy 69.28%로, 목표 80%에 미달한 상태이다.

### 핵심 결론

1. **기존 모델 아키텍처를 폐기하고 새로 설계**한다. 현재 segmented video 기반 단일 시점 모델로는 데이터 한계를 극복할 수 없다.
2. **3개 시점(정면/좌측/우측) 비디오를 하나의 데이터 단위로 통합**하여, 3D Skeleton Triangulation을 수행한 후 이를 모델 입력으로 사용한다. Multi-view triangulation으로 recognition error 50% 이상 감소가 보고되었다 [2].
3. **2단계 계층적 분류(Hierarchical Classification)** 전략을 채택한다: 보행 가능 여부로 1차 분류 → 세부 Level 2차 분류. 이는 GMFCS-ER의 임상 판정 체계와 일치한다 [10].
4. **핵심 동작 4개**(seated_to_standing, crawl, walk, standing_to_seated)로 데이터셋을 구성하고, side_rolling은 비보행 그룹 변별 보조 동작으로 활용한다.
5. **정적 seated(앉아있기)는 변별력이 없으므로 제외**한다 — 소아재활 전문의도 정적 앉기 영상만으로는 Level 구분이 불가하다는 임상적 피드백 확인.

---

## 2. 원본 비디오 → 데이터 샘플 편집 가이드

### 2.1 편집 전 필수 선행 작업: 시간 동기화

원본 비디오는 환자당 3개 카메라(정면 GoPro / 좌측 iPhone / 우측 Galaxy)로 동시 촬영되었으나 녹화 시작 시점이 다르다. **반드시 파이널 컷 프로의 "클립 동기화" 기능(오디오 파형 기반)으로 3개 영상을 시간 정렬한 후에 동작별 컷을 진행**해야 한다. 시각 기반 동기화의 최신 연구로는 VisualSync [7]가 50ms 이하의 동기화 오차를 달성하였다.

```
[필수 워크플로우]
원본_FV.MP4 ─┐
원본_LV.MOV ─┼→ 파이널컷 오디오 동기화 → 동일 시점에서 3개 동시 컷 → triplet 저장
원본_RV.mp4 ─┘
```

### 2.2 프레임레이트 통일

| 카메라 | 원본 FPS | 변환 |
|--------|----------|------|
| GoPro (정면) | ~60fps | → **30fps** |
| iPhone (좌측) | ~60fps | → **30fps** |
| Galaxy (우측) | 30fps | 유지 |

Galaxy가 30fps이므로 나머지를 30fps로 다운샘플한다. 파이널 컷 프로에서 프로젝트 설정을 30fps로 하면 자동 적용된다.

### 2.3 핵심 동작 및 클립 조건

| 우선순위 | 동작 | 클립 길이 | 자르기 기준 | 비고 |
|---------|------|-----------|------------|------|
| **1순위** | **seated_to_standing** | 3~6초 | 앉은 자세에서 일어서기 시작 ~ 완전히 선 후 2초 | 5/5 Level, 19명, 전환 동작의 질이 Level별로 극명히 다름 |
| **2순위** | **crawl** | 5~8초 | 기기 시작 ~ 3회 이상 반복 | 5/5 Level, 15명, L3-L5 변별 핵심 |
| **3순위** | **walk** | 5~8초 | 보행 시작 ~ 방향전환 포함, 편도 or 왕복 | L1-L4, 16명, L5 불가 자체가 정보 |
| **4순위** | **standing_to_seated** | 3~6초 | 서 있다가 앉기까지 전 과정 | 5/5 Level, 15명, 전환 동작 |
| 보조 | **side_rolling** | 5~8초 | 옆구르기 2~3회 | L4-L5만, 비보행 그룹 변별 보조 |
| ~~제외~~ | ~~seated (정적 앉기)~~ | — | — | 전문의도 Level 구분 불가 |
| ~~제외~~ | ~~run, jump~~ | — | — | L1-L2 전용, shortcut 학습 유발 |

### 2.4 한 클립의 원칙

- **한 클립 = 한 동작 유형**: 걷다가 방향 전환은 같은 클립에 포함해도 됨 (walk은 walk). 걷다가 앉는 경우는 별도 클립으로 분리 (walk ≠ standing_to_seated).
- **한 클립 = 3개 시점 동시 컷**: FV/LV/RV를 동일한 시간 구간에서 동시에 잘라야 함.
- **프레임 수 일치 확인 필수**: 3개 클립의 duration 차이가 ±0.1초 이내여야 함.

### 2.5 Occlusion(가림) 처리 기준

| 상황 | 판단 | 이유 |
|------|------|------|
| 1개 시점에서 일시적(1~2초) 가림 | **포함** | 나머지 2시점으로 3D 복원 가능 |
| 1개 시점에서 장시간(절반 이상) 가림 | **포함하되 표시** | 자동 처리로 가중치 조절 |
| 2개 시점 동시 가림 | **해당 구간 제외** | 1시점만으로는 3D 불가 |
| 3개 시점 모두 가림 | **해당 구간 제외** | 정보 없음 |

### 2.6 파일명 및 디렉토리 구조

```
CP_dataset/
├── raw_synced/
│   ├── ajy/
│   │   ├── ajy_seated_to_standing_01_FV.mp4
│   │   ├── ajy_seated_to_standing_01_LV.mp4
│   │   ├── ajy_seated_to_standing_01_RV.mp4    ← 이 3개가 하나의 triplet
│   │   ├── ajy_crawl_01_FV.mp4
│   │   ├── ajy_crawl_01_LV.mp4
│   │   ├── ajy_crawl_01_RV.mp4
│   │   └── ...
│   └── ...
├── skeleton_2d/          # 2D pose estimation 결과
├── calibration/          # 환자별 카메라 파라미터
├── skeleton_3d/          # 3D triangulation 결과 (.npy)
└── metadata/
    ├── labels.json       # GMFCS 레이블
    └── triplets.json     # triplet 매핑 정보
```

**네이밍 규칙**: `{환자ID}_{동작}_{번호}_{시점}.mp4`
- `{환자ID}_{동작}_{번호}` = triplet 식별자 (3개 시점이 이 부분을 공유)
- `_{시점}` = FV / LV / RV

---

## 3. 데이터셋 현황

### 3.1 환자 정보 (24명)

| 항목 | 값 |
|------|-----|
| 총 환자 수 | 24명 |
| 연령 | 평균 47.2개월 (23~72개월) |
| 성별 | 남 14명, 여 10명 |
| CP 유형 | Spastic diplegia 13명(54.2%), Hemiplegia 5명, Quadriplegia 3명, Dyskinetic 2명, Other 1명 |
| 의사/보호자 GMFCS 일치율 | 100% (24명 전원 일치) |

### 3.2 GMFCS Level별 분포

| GMFCS | 환자 수 | 클립 수 | 비율 | 환자당 평균 클립 |
|-------|---------|---------|------|-----------------|
| Level 1 | 6명 | 1,099 | 34.6% | 183.2 |
| Level 2 | 5명 | 783 | 24.7% | 156.6 |
| Level 3 | 4명 | 595 | 18.7% | 148.8 |
| Level 4 | 3명 | 360 | 11.3% | 120.0 |
| Level 5 | 6명 | 338 | 10.6% | 56.3 |
| **합계** | **24명** | **3,175** | **100%** | — |

### 3.3 클래스 불균형 심각도

- Level 1(1,099) vs Level 5(338): **3.25배** 차이
- Level 4: 환자 **3명**뿐 (hdi, jrh, lsa)
- hdu(Level 5): 클립 **6개**만 보유
- kra(Level 1): 클립 **292개** → hdu와 약 **49배** 차이

### 3.4 Complete Triplet (FV+LV+RV) 현황

| 항목 | 수치 |
|------|------|
| Complete triplets | 868개 |
| Incomplete | 362개 |
| Triplet coverage | 2,604 / 3,175 클립 (82.0%) |

---

## 4. 촬영 환경

### 4.1 카메라 배치 (촬영 세팅 문서 기준)

| 위치 | 장비 | 각도 | 높이 | 거리 |
|------|------|------|------|------|
| 정면 (FV) | GoPro | 0° | 90cm | 250cm |
| 좌측 (LV) | iPhone (12 mini 등) | 45° | 90cm | 250cm |
| 우측 (RV) | Samsung Galaxy | 30° (기존 45°에서 변경) | 90cm | 250cm |

**주의**: 실제 촬영에서 연구원이 이 조건을 정확히 지키지 않아 환자별로 상당한 편차가 존재함.

### 4.2 카메라별 사양

| 카메라 | 해상도 | FPS | 코덱 | 특이사항 |
|--------|--------|-----|------|----------|
| GoPro (정면) | 1920×1080 | ~60fps | H.264/H.265 | 광각 왜곡(barrel distortion) 보정 필요 |
| iPhone (좌측) | 1920×1080 | ~60fps | H.264 | 일부 세로 촬영(rotate=90) |
| Galaxy (우측) | 1920×1080 또는 1080×2320 | **30fps** | HEVC | 세로 모드 영상 → 연구에서 제외 |

### 4.3 촬영 장소

삼성서울병원 소아 물리치료실 (단일 고정 공간). 바닥에 녹색 매트 배치, 주변에 재활 장비 존재. 촬영 세팅은 Kim et al. [11]의 K-DST 기반 다중 카메라 소아 발달 평가 프로토콜을 참고하였다.

### 4.4 캘리브레이션 데이터

**캘리브레이션 파일 없음**. checkerboard 촬영, 카메라 intrinsic/extrinsic 파라미터 파일 모두 존재하지 않음. 다만 "Human Pose as Calibration Pattern" 기법 [5]으로 영상 내 아이의 관절을 대응점으로 사용하여 사후 캘리브레이션이 가능함. Liu et al. [4]과 Pätzold et al. [6]도 유사하게 사람 키포인트를 이용한 자동 캘리브레이션의 효과를 입증하였다.

---

## 5. 현재 모델 성능 (중간결과)

### 5.1 학습 환경

| 항목 | 사양 |
|------|------|
| 서버 | 삼성서울병원 |
| GPU | NVIDIA Tesla V100-DGXS-32GB × 2 |
| CUDA | 12.4 |
| 디스크 | 1.5TB 가용 |
| 입력 데이터 | SAM2 Segmented Video |
| 학습 현황 | Epoch 6/20, tmux main session에서 실행 중 |

### 5.2 Validation 결과 (Epoch 6/20, Val Loss: 1.3092)

| GMFCS Level | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Level I | 0.67 | 0.98 | 0.80 | 118 |
| Level II | 0.88 | 0.77 | 0.82 | 90 |
| Level III | 0.69 | 0.64 | 0.66 | 152 |
| Level IV | 1.00 | **0.20** | 0.33 | 30 |
| Level V | 0.52 | 0.48 | 0.50 | 82 |

| 종합 지표 | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Overall Accuracy | — | — | **69.28%** |
| Macro Avg | 0.75 | 0.61 | 0.62 |
| Weighted Avg | 0.71 | 0.69 | 0.68 |

### 5.3 현재 모델의 문제점 분석

- **Level IV**: Recall 0.20 → 30개 중 6개만 정확 분류, 나머지 24개 오분류 (심각한 과소탐지)
- **Level V**: 전반적 저성능 (F1 0.50), 동작 패턴이 seated/side_rolling으로 제한적
- **Level I**: Recall 0.98이지만 Precision 0.67 → 다른 Level을 Level I로 오분류하는 경향
- **Level III**: support 152로 가장 많으나 F1 0.66 → 중간 Level의 모호성

---

## 6. 동작(Movement) 분석

### 6.1 동작별 GMFCS Level Coverage (실제 데이터)

| 동작 | L1 | L2 | L3 | L4 | L5 | Levels | 환자 수 |
|------|----|----|----|----|----|----|---------|
| **seated** | 5명 | 5명 | 4명 | 3명 | 4명 | 5/5 | 20명 |
| **seated_to_standing** | 5명 | 5명 | 3명 | 2명 | 3명 | 5/5 | 19명 |
| **standing_to_seated** | 3명 | 3명 | 3명 | 2명 | 1명 | 5/5 | 15명 |
| **crawl** | 1명 | 2명 | 2명 | 3명 | 2명 | 5/5 | 15명 |
| **walk** | 4명 | 5명 | 3명 | 2명 | 0명 | 4/5 | 16명 |
| side_rolling | 0명 | 0명 | 0명 | 2명 | 5명 | 2/5 | 7명 |
| run | 5명 | 1명 | 0명 | 0명 | 0명 | 2/5 | 6명 |
| jump | 4명 | 2명 | 0명 | 0명 | 0명 | 2/5 | 6명 |

### 6.2 핵심 동작별 Triplet 수

| 동작 | L1 | L2 | L3 | L4 | L5 | 합계 |
|------|----|----|----|----|----|----|
| seated | 81 | 52 | 37 | 43 | 47 | 260 |
| seated_to_standing | 22 | 24 | 15 | 17 | 4 | 82 |
| crawl | 6 | 5 | 37 | 37 | 15 | 100 |
| walk | 83 | 60 | 31 | 12 | 0 | 186 |
| standing_to_seated | 13 | 17 | 9 | 14 | 2 | 55 |
| side_rolling | 0 | 0 | 0 | 4 | 37 | 41 |

### 6.3 동작 제외 이유 및 결정

**제외 대상 (shortcut 학습 유발)**:

| 동작 | 제외 이유 |
|------|-----------|
| side_rolling | 95%가 L5, "옆구르기=L5" shortcut 학습 유발 (단, 비보행 그룹 보조 동작으로 활용) |
| run | 76%가 L1, "달리기=경증" 학습 |
| jump | 83%가 L1, "점프=경증" 학습 |
| seated (정적) | **소아재활 전문의도 Level 구분 불가** — L1~L4 모두 안정적으로 앉아있음 |
| seated_to_chair_seated | 96%가 L3 전용 |
| prone_to_seated | 95%가 L4 전용, 2명만 |
| 기타 극소 동작 | walk_to_chair, supine_to_prone, crawl_to_seated, roll_side — 모두 1명 전용 |

### 6.4 Level 4, 5 환자의 실제 수행 가능 동작

**Level 4:**
- hdi: crawl(32), seated(27), walk(25), seated_to_standing(15), standing_to_seated(15), side_rolling(6)
- jrh: crawl(75), seated(42), seated_to_standing(39), standing_to_seated(30), walk(21)
- lsa: seated(15), crawl(9), side_rolling(6) — **walk 불가, seated_to_standing 불가**

**Level 5:**
- ajy: seated(50), side_rolling(6) — walk/crawl 불가
- hdu: side_rolling(6) — 데이터 극소
- kcw: side_rolling(41) — side_rolling만 보유
- kri: crawl(51), seated(15), seated_to_standing(3) — **crawl 가능한 L5**
- oms: seated(45), side_rolling(36), seated_to_standing(3)
- pjo: side_rolling(27), seated(23), crawl(12), seated_to_standing(6), standing_to_seated(6)

---

## 7. 보행 가능 여부에 의한 계층적 분류 전략

### 7.1 Walk 가능 여부 데이터 확인

| Level | 환자 | Walk 가능 | Walk 불가 |
|-------|------|-----------|-----------|
| L1 | 6명 | **6명 전원** | 0명 |
| L2 | 5명 | **5명 전원** | 0명 |
| L3 | 4명 | ly, mkj, pjw (3명) | **kku** (1명) |
| L4 | 3명 | hdi, jrh (2명) | **lsa** (1명) |
| L5 | 6명 | 0명 | **6명 전원** |

**Walk 가능 16명 / Walk 불가 8명**

### 7.2 2단계 계층적 분류 구조

```
                       모든 환자 (24명)
                            │
                  ━━━━━━━━━━┿━━━━━━━━━━
                  │                    │
            [Stage 1]            [Stage 1]
           Ambulatory           Non-ambulatory
          walk 가능 그룹        walk 불가 그룹
         (16명, ~2,600클립)    (8명, ~560클립)
                  │                    │
            ┌─────┼─────┐        ┌─────┼─────┐
            │     │     │        │           │
         [Stage 2-A]          [Stage 2-B]
         L1   L2   L3(L4)    L3(L4)     L5

         walk + s2s 질로     crawl + side_rolling
         세부 변별              질로 세부 변별
```

### 7.3 Stage별 사용 동작

| Stage | 분류 목표 | 사용 동작 | 변별 포인트 |
|-------|-----------|-----------|-------------|
| Stage 1 | Ambulatory vs Non-ambulatory | seated_to_standing, crawl | 전환 동작의 질, 독립 수행 가능 여부 |
| Stage 2-A | L1 vs L2 vs L3-L4 | walk, seated_to_standing | 보행 속도/대칭성, 전환 동작 안정성 |
| Stage 2-B | L3-L4 vs L5 | crawl, side_rolling, seated_to_standing | 기기 가능 여부, 이동 수단 |

### 7.4 이 전략의 장점

1. **임상 판정 과정과 동일**: 의사도 "걸을 수 있는가?" → "얼마나 잘 걷는가?" 순서로 판정 [10]
2. **클래스 불균형 완화**: 5-class 한 번 → binary + 3-class 두 번으로 분할
3. **각 단계에 최적 동작 투입**: Stage마다 가장 변별력 높은 동작만 사용
4. **80% 달성 가능성**: Stage 1에서 90%, Stage 2에서 85% 달성 시 전체 ~76%, 3D Triangulation + multi-view 추가로 80%+ 가능

---

## 8. 3D Skeleton Triangulation 전략

### 8.1 왜 3D인가

현재 모델은 단일 시점 2D 영상을 입력으로 사용한다. 이 경우 depth ambiguity (깊이 모호성)으로 인해 관절의 실제 3D 위치를 알 수 없다. 3개 시점을 triangulation하면:
- 관절의 실제 3D 좌표 (x, y, z)를 복원
- 단일 시점 occlusion 문제 해결 (다른 시점에서 보완)
- View-invariant feature 직접 제공
- Skarimva [2]에서 multi-view triangulation으로 recognition error 50%+ 감소 보고. Dual-Camera CP Gait Analysis [3]에서도 3D 재구성을 통한 CP 보행 분석의 정확도 향상을 확인

### 8.2 자동 캘리브레이션 파이프라인

캘리브레이션 파일이 없으므로, Takahashi et al. [5]이 제안한 "Human Pose as Calibration Pattern" 기법을 사용한다:

```
[Step 1] 시간 동기화
    → 파이널 컷 프로의 오디오 기반 자동 동기화 (VisualSync [7] 등 시각 기반 동기화도 가능)

[Step 2] 2D Pose Estimation (각 시점별)
    → MediaPipe/OpenPose로 아이의 2D 관절 좌표 추출

[Step 3] 카메라 Intrinsic 추정
    → GoPro/iPhone/Galaxy 기종별 알려진 focal length 활용
    → GoPro 광각 왜곡 보정

[Step 4] 카메라 Extrinsic 자동 추정
    → 동기화된 프레임의 2D 관절 좌표를 대응점으로 사용
    → Fundamental Matrix → Essential Matrix → Rotation/Translation 복원 [4][6]
    → 환자(촬영 세션)별 독립 추정 (촬영 조건 편차 자동 해결)
    → SteerPose [12] 등 최신 neural 기반 캘리브레이션도 적용 가능

[Step 5] 3D Triangulation
    → OpenCV triangulatePoints로 3D 관절 좌표 복원
    → 출력: (T, 17, 3) — 프레임 × 관절 × xyz
```

### 8.3 촬영 조건 편차 문제 해결

연구원이 촬영 세팅(250cm, 90cm, 각도)을 정확히 지키지 않았으므로 환자별로 카메라 위치가 크게 다를 수 있다. 그러나 Human Pose as Calibration Pattern [5]은 **영상 자체의 내재적 특성**에서 카메라 파라미터를 추정하므로, 촬영 세팅 문서의 숫자에 의존하지 않는다. Liu et al. [4]은 이 방식이 전통적 체커보드 캘리브레이션보다 우수한 성능을 보임을 입증했고, Pätzold et al. [6]은 수 분 내에 수렴하는 온라인 캘리브레이션을 시연하였다. 환자별로 독립 추정하기 때문에 **오히려 고정값을 사용하는 것보다 더 정확**하다.

---

## 9. Multi-view Score Fusion에 대한 검토

### 9.1 원래 검토한 방법

```
P_final = (P_FV + P_LV + P_RV) / 3
GMFCS_predicted = argmax(P_final)
```

### 9.2 한계: 개별 시점 성능이 낮으면 평균도 낮다

이 방법은 3개 view가 서로 **다른 종류의 오류**를 범할 때만 보정 효과가 있다 [8]. 현재 모델처럼 Level IV를 거의 예측하지 못하는(Recall 0.20) 상황에서는 FV든 LV든 RV든 전부 같은 방향으로 틀릴 가능성이 높다. Score Fusion은 이미 충분히 좋은 모델의 마지막 2~3%를 끌어올리는 "마무리 전략"이지, 근본적으로 성능이 부족한 모델을 살리는 방법이 아니다.

→ 따라서 **Multi-view 정보를 모델 내부에서 처음부터 함께 학습하는 아키텍처** (3D Triangulation → 3D skeleton 입력)가 필요하다.

---

## 10. 데이터 샘플 길이 및 구성 검토

### 10.1 클립 길이에 대한 4가지 전략 검토

| 전략 | 판단 | 이유 |
|------|------|------|
| 클립 복사(looping)로 길이 두 배 | **비권장** | 새 정보 없음, temporal discontinuity, spurious correlation 과적합 |
| 같은 동작 2개 클립 temporal concatenation | **조건부 권장** | 같은 동작 유형일 때만, 반복 패턴의 일관성/변동성 학습 |
| 3개+ 클립 물리적 합성 | **비권장** | Clip-level Aggregation (embedding voting)이 더 효과적 |
| **3개 시점을 하나의 단위로** | **강력 권장** | 3D Triangulation으로 근본적 정보량 증가 |

### 10.2 현재 클립 통계

| 항목 | 값 |
|------|-----|
| 전체 평균 길이 | 8.0초 (중앙값 7.3초) |
| 범위 | 0.9초 ~ 20.1초 |
| walk 평균 | 5.4초 |
| crawl 평균 | 5.0초 |
| seated 평균 | 8.9초 |
| seated_to_standing 평균 | 6.3초 |
| side_rolling 평균 | 12.0초 |

---

## 11. 선행 연구 (Zhao et al., IEEE TNSRE 2024)

### 11.1 개요

Zhao et al. [1]은 STGCN + metric learning (triplet loss + consistency loss)으로 GMFCS I~IV 분류를 수행하였다. MIT 팀, Kidzinski 공개 데이터셋 사용. STGCN의 backbone은 NTU RGB+D [9] 데이터셋으로 사전학습되었다.

### 11.2 핵심 수치

| 항목 | 값 |
|------|-----|
| 데이터셋 | 861명, 1,450비디오 |
| 대상 연령 | 평균 11세 (s.d. 5.9) |
| GMFCS 범위 | Level I~IV (Level V 제외) |
| Pose Estimation | OpenPose |
| Encoder | STGCN (NTU RGB+D 120 [9]으로 transfer learning) |
| End-to-end 정확도 | 76.6% |
| Metric learning + confidence 0.95 | **88%** |
| Cohen's Kappa | κlw = 0.733 |
| Train/Val/Test split | 환자 단위 7:1:2 |

### 11.3 본 연구와의 비교

| 항목 | Zhao et al. | 본 연구 |
|------|-------------|---------|
| 환자 수 | 861명 | **24명** |
| 대상 연령 | 평균 11세 | **6세 이하** |
| GMFCS Level | I~IV | **I~V** |
| 카메라 시점 | 단일 | **3개 (FV/LV/RV)** |
| 입력 | 2D skeleton | **3D skeleton (목표)** |
| 동작 | 걷기/달리기 | **다중 동작** |

---

## 12. 서버 환경

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA Tesla V100-DGXS-32GB × 2 (현재 거의 미사용) |
| CUDA | 12.4 |
| 드라이버 | 550.144.03 |
| 디스크 전체 | 1.8TB |
| 디스크 가용 | 1.5TB (현재 158GB 사용, 10%) |
| 서버 접속 | http://14.63.89.203:28888/lab |

---

## 13. 핵심 문제 요약 및 해결 전략

| 문제 | 원인 | 해결 전략 |
|------|------|-----------|
| 절대적 샘플 부족 (24명) | 선행 연구 861명 대비 극소 [1] | 3D Triangulation으로 정보량 극대화 [2][3], 계층적 분류로 단계별 난이도 감소 |
| 클래스 불균형 | L1(1,099) vs L5(338), L4(3명) | 2단계 분류로 binary → 소분류, class weight 조절 |
| 동작 유형 confounding | "side_rolling=L5" shortcut | 핵심 4개 동작만 사용, 다수 Level에 걸친 동작만 유지 |
| L1~5 공통 동작 부재 | L5는 walk 불가, L1은 side_rolling 불필요 | 계층적 분류: "할 수 있는가" → "얼마나 잘 하는가" |
| 정적 seated 변별력 부재 | 전문의도 구분 불가 | seated 정적 구간 제외, 동적 전환 동작에 집중 |
| 환자 단위 data leakage | 같은 환자 클립이 train/test에 걸침 | 반드시 환자 단위 split |
| 단일 시점 한계 | depth ambiguity, occlusion | 3D Triangulation [2][3] + Human Pose Calibration [4][5][6] |

---

## 14. 원본 비디오 기반 환자별 동작 수행 현황 (실측 데이터)

원본 비디오 디렉토리(`CP_videos_cut_original/data/raw/`)의 파일을 직접 전수 조사하여, 24명 환자 × 5개 핵심 동작의 정확한 클립 수를 산출하였다. `chair_seated_to_standing` 등 유사 명칭 동작과의 혼동을 배제하기 위해 exact string matching을 적용하였다.

### 14.1 L1 환자 (6명) 동작 수행 현황

| 동작 | jyh | kdu | kra | kto | orj | phm | 수행률 | 총 클립 |
|------|-----|-----|-----|-----|-----|-----|--------|---------|
| **walk (w)** | 23 | 80 | 85 | 98 | 14 | 29 | **6/6** | **329** |
| **seated_to_standing (c_s)** | 6 | 24 | 3 | 33 | 15 | 14 | **6/6** | **95** |
| **standing_to_seated (s_c)** | 2 | 22 | ✗ | 19 | 15 | 4 | 5/6 | 62 |
| crawl (cr) | 14 | ✗ | ✗ | 21 | 8 | 2 | 4/6 | 45 |
| side_rolling (sr) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0/6 | 0 |

### 14.2 L2 환자 (5명) 동작 수행 현황

| 동작 | hja | jeu | jji | jty | lrl | 수행률 | 총 클립 |
|------|-----|-----|-----|-----|-----|--------|---------|
| **walk (w)** | 35 | 45 | 24 | 27 | 78 | **5/5** | **209** |
| **seated_to_standing (c_s)** | 21 | 8 | 21 | 6 | 35 | **5/5** | **91** |
| **standing_to_seated (s_c)** | 21 | 2 | 13 | ✗ | 33 | 4/5 | 69 |
| crawl (cr) | 3 | 14 | 2 | ✗ | ✗ | 3/5 | 19 |
| side_rolling (sr) | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 | 0 |

### 14.3 L3 환자 (4명) 동작 수행 현황

| 동작 | kku | ly | mkj | pjw | 수행률 | 총 클립 |
|------|-----|-----|-----|-----|--------|---------|
| **crawl (cr)** | 91 | 27 | ✗ | 2 | **3/4** | **120** |
| **walk (w)** | ✗ | 11 | 53 | 34 | **3/4** | **98** |
| **seated_to_standing (c_s)** | 13 | 15 | ✗ | 24 | **3/4** | **52** |
| **standing_to_seated (s_c)** | 7 | 8 | ✗ | 15 | 3/4 | 30 |
| side_rolling (sr) | ✗ | ✗ | ✗ | ✗ | 0/4 | 0 |

### 14.4 L4 환자 (3명) 동작 수행 현황

| 동작 | hdi | jrh | lsa | 수행률 | 총 클립 |
|------|-----|-----|-----|--------|---------|
| **crawl (cr)** | 32 | 75 | 9 | **3/3 (100%)** | **116** |
| **walk (w)** | 25 | 21 | ✗ | 2/3 | 46 |
| **seated_to_standing (c_s)** | 15 | 39 | ✗ | 2/3 | 54 |
| **standing_to_seated (s_c)** | 15 | 30 | ✗ | 2/3 | 45 |
| **side_rolling (sr)** | 6 | ✗ | 6 | 2/3 | 12 |

### 14.5 L5 환자 (6명) 동작 수행 현황

| 동작 | ajy | hdu | kcw | kri | oms | pjo | 수행률 | 총 클립 |
|------|-----|-----|-----|-----|-----|-----|--------|---------|
| **side_rolling (sr)** | 6 | 6 | 41 | ✗ | 36 | 27 | **5/6 (83%)** | **116** |
| crawl (cr) | ✗ | ✗ | ✗ | 51 | ✗ | 12 | 2/6 | 63 |
| seated_to_standing (c_s) | ✗ | ✗ | ✗ | 3 | 3 | 6 | 3/6 | 12 |
| standing_to_seated (s_c) | ✗ | ✗ | ✗ | ✗ | ✗ | 6 | 1/6 | 6 |
| walk (w) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0/6 | 0 |

---

## 15. GMFCS Level 간 변별 동작 심층 분석

### 15.1 L4 vs L5 변별 분석

L4와 L5는 모두 비보행(Non-ambulatory) 그룹에 속하며, 가장 중증인 두 레벨 사이의 변별이 필요하다.

**동작별 L4∩L5 교집합:**

| 동작 | L4 수행 환자 | L5 수행 환자 | 양쪽 모두? | 변별력 |
|------|-------------|-------------|-----------|--------|
| **crawl (cr)** | 3/3 (100%) | 2/6 (33%) | △ 부분적 | ⭕ **crawl 가능 여부 자체가 L4→L5 분기 신호** |
| **side_rolling (sr)** | 2/3 (67%) | 5/6 (83%) | △ 부분적 | ⭕ 구르기 질적 차이 (대칭성, 속도, 자발성) |
| **seated_to_standing (c_s)** | 2/3 (67%) | 3/6 (50%) | △ 부분적 | ⭕ 전환 능력의 질적 차이 |
| standing_to_seated (s_c) | 2/3 (67%) | 1/6 (17%) | ❌ 거의 없음 | — |
| walk (w) | 2/3 (67%) | 0/6 (0%) | ❌ L5 불가 | — |

**변별 전략:** crawl 1순위 (L4 전원 가능 vs L5 대부분 불가), side_rolling 2순위 (질적 차이), seated_to_standing 3순위 (보조).

**주의:** side_rolling만으로는 부족하다 — L4의 jrh는 side_rolling 데이터가 없고, L5의 kri는 side_rolling 데이터가 없다.

### 15.2 L1 vs L2 vs L3 변별 분석

L1, L2, L3는 보행 가능(Ambulatory) 그룹에 속하며, 보행의 질적 차이로 세분류한다.

**동작별 3개 레벨 교집합:**

| 동작 | L1 수행률 | L2 수행률 | L3 수행률 | 3레벨 모두? | 변별력 |
|------|-----------|-----------|-----------|-------------|--------|
| **walk (w)** | 6/6 (100%) | 5/5 (100%) | 3/4 (75%) | ⭕ **최우선** | L1 빠르고 안정 → L2 약간 비대칭 → L3 느리고 불안정·보조기구 |
| **seated_to_standing (c_s)** | 6/6 (100%) | 5/5 (100%) | 3/4 (75%) | ⭕ **최우선** | L1 즉각 독립 → L2 약간 지연 → L3 손짚기·보조 필요 |
| **standing_to_seated (s_c)** | 5/6 (83%) | 4/5 (80%) | 3/4 (75%) | △ 양호 | 앉는 과정 제어력 차이 (보조) |
| crawl (cr) | 4/6 (67%) | 3/5 (60%) | 3/4 (75%) | △ 부분적 | L1·L2는 불필요(걸으면 되니까), L3는 주력 이동수단 |
| side_rolling (sr) | 0/6 | 0/5 | 0/4 | ❌ 해당 없음 | — |

**핵심:** L1 vs L2는 동작 수행 패턴이 거의 동일하므로 **영상의 질적 차이(보행 속도, 대칭성, 안정성)**로만 구분 가능하다. 메타데이터만으로는 L1과 L2를 분리할 수 없다.

### 15.3 L3 vs L4 변별 분석 — 🔑 교수님 핵심 관심 영역

GMFCS에서 L3과 L4의 경계는 **"독립 보행 가능 vs 보조 이동수단 의존"**의 분기점이며 [10], 논문 지도교수님이 가장 핵심적인 구분으로 지목한 영역이다.

**동작별 L3∩L4 교집합:**

| 동작 | L3 수행 | L4 수행 | 양쪽 모두? | 변별력 | 클립 합계 |
|------|---------|---------|-----------|--------|-----------|
| **crawl (cr)** | 3/4 (75%) | **3/3 (100%)** | ⭕ | ⭕⭕⭕ **최고** | 236 |
| **walk (w)** | 3/4 (75%) | 2/3 (67%) | ⭕ | ⭕⭕ 높음 | 144 |
| **seated_to_standing (c_s)** | 3/4 (75%) | 2/3 (67%) | ⭕ | ⭕⭕ 높음 | 106 |
| **standing_to_seated (s_c)** | 3/4 (75%) | 2/3 (67%) | △ | ⭕ 보조 | 75 |
| side_rolling (sr) | 0/4 | 2/3 | ❌ L3 없음 | — | — |

**crawl이 L3 vs L4 변별의 핵심 동작인 이유:**

1. **Coverage 최고**: L3 3/4(120클립) + L4 3/3(116클립) = 236클립, 양쪽 모두 가장 높은 수행률
2. **질적 차이 극명**: L3은 reciprocal crawl(교대 사지 패턴, 리드미컬) / L4는 불완전 교대 패턴, 배밀이에 가까움, 비대칭
3. **클립 duration 차이**: L3 평균 3.4~5.6초 vs L4 평균 4.5~8.5초 → L4가 유의하게 느림
4. **lsa(L4) 포함 가능**: walk 불가인 lsa도 crawl 9개를 수행 → walk 기반 비교 불가 환자도 포함

**L3 vs L4 변별의 어려운 점:**

- **lsa(L4, 23개월)**: walk, seated_to_standing, standing_to_seated 전부 불가. 사실상 L5에 가까운 기능 프로파일
- **mkj(L3, 60개월)**: crawl, seated_to_standing, standing_to_seated 데이터 없음. 의자 관련 동작과 walk만 수행
- **hdi·jrh(L4)**: walk 가능 (hdi 25클립, jrh 21클립) → L3과의 경계가 모호

### 15.4 GMFCS-E&R 기반 동작 질적 차이 기술서 (Movement Quality Descriptors)

현재까지의 분석은 **"어떤 동작을 수행할 수 있는가"** (가능/불가)에 집중하였다. 그러나 L1 vs L2, L3 vs L4처럼 동일한 동작을 수행하면서도 Level이 다른 경우, **동작 수행의 질적 차이**만이 분류 단서가 된다. GMFCS-E&R [10]의 연령대별 기능 기술(Age Band Descriptions)과 Level 간 구분점(Distinctions Between Levels)을 기반으로, 본 프로젝트의 대상 연령(23~72개월)에 해당하는 3개 연령대에서 핵심 동작별 질적 차이를 정리한다.

#### 15.4.1 Walk (보행) 질적 차이 — L1 vs L2 vs L3 vs L4

| 질적 특성 | L1 | L2 | L3 | L4 |
|-----------|-----|-----|-----|-----|
| **보행 독립성** | 보조기구 없이 독립 보행, 계단 포함 | 보조기구 없이 실내 보행 가능하나 장거리·불균형 표면 제한 | hand-held mobility device (walker) 필요 | 성인 보조 + walker, 단거리만 |
| **속도·리듬** | 빠르고 일정한 cadence, 달리기·점프 능력 출현 | 느림, cadence 불규칙, 달리기·점프 불가 | 현저히 느림, 보조기구에 의존한 리듬 | 극히 느림, 보행보다 휠체어 선호 |
| **대칭성** | 좌우 대칭적 swing phase | 약간의 비대칭 (hemiplegic 패턴 가능) | 뚜렷한 비대칭, 보상적 체간 측굴 | 심한 비대칭, 양측 모두 기능 저하 |
| **균형 유지** | 안정적, 방향전환 시에도 유지 | 장거리 보행·불균형 표면에서 균형 상실 | 보조기구 없이 균형 유지 불가 | 체중 지지 불안정, 균형 극도로 제한 |
| **계단 보행** | 독립적 계단 보행 출현 (4~6세) | 난간 사용 필수 | 성인 보조 필수 | 불가 |
| **상지 역할** | 자유 (물건 들기 가능) | 간헐적 균형 보조로 사용 | 보조기구 파지에 사용, 자유 상지 없음 | 보조기구 + 성인 보조 |

**Skeleton 기반 측정 가능 특성:**
- **Cadence (보폭 주기):** ankle joint의 주기적 x-축 변위로 계산 → L1 > L2 > L3 > L4
- **Gait symmetry index:** 좌우 ankle swing duration 비율 → L1 ≈ 1.0, L2 < 1.0, L3 ≪ 1.0
- **Trunk lateral sway:** pelvis/spine의 좌우 흔들림 진폭 → L3, L4에서 현저히 증가
- **Step width:** 양측 ankle 간 거리 → 중증일수록 넓은 base of support
- **Upper limb freedom:** wrist 좌표의 자유 운동 범위 → L1은 자유로움, L3 이상은 고정됨 (보조기구 파지)
- **Head stability:** head joint의 vertical/lateral oscillation → 중증일수록 불안정

#### 15.4.2 Seated-to-Standing (앉기→서기 전환) 질적 차이 — 전 Level

| 질적 특성 | L1 | L2 | L3 | L4 | L5 |
|-----------|-----|-----|-----|-----|-----|
| **전환 소요 시간** | 즉각 (<1초) | 약간의 지연 (1~2초) | 느림 (2~4초), 중간에 정지 가능 | 매우 느림 (>4초), 성인 보조 or 잡을 곳 필수 | 독립 수행 불가 (극소수 매우 제한적 시도) |
| **손 사용** | 손 사용 없이 일어남 | 때때로 바닥이나 무릎에 손 짚음 | 안정된 표면(의자, 바닥)에 손 짚기 필수 | 성인 보조 or 고정 물체 필수 | — |
| **체간 제어** | 직립 자세로 일관되게 전환 | 약간의 전방 기울임 후 교정 | 현저한 전방 기울임, 체간 굴곡 보상 | 체간 제어 불충분, 흔들림 심함 | — |
| **하지 대칭** | 양 하지 동시·대칭적 | 약간의 비대칭 | 비대칭적, 한 쪽 하지 선도 | 심한 비대칭, 한 쪽 하지 의존 | — |
| **균형 회복** | 선 후 즉각 안정 | 선 후 약간의 흔들림 | 선 후 수 초간 불안정, 보조기구 파지 | 선 후에도 독립 유지 불가 | — |

**Skeleton 기반 측정 가능 특성:**
- **Transition duration:** hip joint가 seated 높이에서 standing 높이에 도달하는 시간 → L1 < L2 < L3 < L4
- **Trunk anterior tilt:** spine-pelvis angle의 최대 전방 기울기 → L3, L4에서 현저
- **Hand-to-ground contact:** wrist joint의 최저점이 floor level에 접근하는 빈도 → L2부터 증가
- **CoM (Center of Mass) trajectory smoothness:** pelvis 궤적의 jerk (3차 미분) → L1은 부드러움, L3+ 불규칙
- **Bilateral knee extension symmetry:** 좌우 knee angle 변화 시차 → L1 동시, L3+ 비대칭적
- **Post-transition sway:** standing 도달 후 2초간 pelvis의 AP/ML oscillation → 중증일수록 큼

#### 15.4.3 Crawl (기기) 질적 차이 — L3 vs L4 vs L5

| 질적 특성 | L3 | L4 | L5 |
|-----------|-----|-----|-----|
| **교대 패턴 (reciprocal)** | ⭕ 교대 사지 패턴 (right hand + left knee → left hand + right knee) | △ 불완전한 교대 패턴, 동측 사지 동시 사용 (homolateral) 경향 | ❌ 교대 불가, belly crawling (배밀이) 또는 수행 불가 |
| **속도·리듬** | 리드미컬하고 일정한 속도 | 불규칙, 멈춤·재시작 빈번 | 극히 느림, 추진력 부족 |
| **체간 높이** | 바닥에서 체간이 충분히 들림 (hands-and-knees) | 체간이 바닥에 가까움, 팔꿈치 기기(commando crawl) 빈번 | 체간을 들 수 없음, 배밀이(belly crawl)만 가능 |
| **체간 회전** | 최소한의 체간 회전, 안정적 | 과도한 체간 회전, 몸통 뒤틀림 | — |
| **이동 거리** | 주력 이동 수단으로 사용, 상당 거리 이동 가능 | 제한된 단거리만 이동 | 극소 거리 또는 이동 불가 |
| **하지 참여** | 양 하지 교대로 적극 참여 | 하지 끌림, 상지 주도형 이동 | 하지 거의 참여 없음 |

**Skeleton 기반 측정 가능 특성:**
- **Reciprocal pattern index:** contralateral hand-knee 쌍의 동시 전진 비율 → L3 ≈ 1.0, L4 < 0.7, L5 ≈ 0
- **Trunk elevation:** pelvis-to-ground 평균 높이 → L3 > L4 ≫ L5 (배밀이)
- **Crawl velocity:** pelvis의 수평 이동 속도 → L3 > L4 > L5
- **Cycle regularity:** hand/knee 전진 주기의 변동계수 (CV) → L3 낮음 (규칙적), L4 높음 (불규칙)
- **Trunk roll amplitude:** spine의 좌우 roll 각도 → L4에서 과도, L3에서 최소
- **Upper vs lower limb contribution ratio:** 상지 이동 거리 / 하지 이동 거리 → L4에서 상지 비율 높음 (상지 주도)

#### 15.4.4 Standing-to-Seated (서기→앉기 전환) 질적 차이 — L1 vs L2 vs L3 vs L4

| 질적 특성 | L1 | L2 | L3 | L4 |
|-----------|-----|-----|-----|-----|
| **하강 제어** | 부드럽고 제어된 하강 | 약간 불안정하나 독립적 | 손 짚기 또는 보조기구 잡기 필수 | 성인 보조 필수, "떨어지듯" 앉음 |
| **하강 속도** | 일정한 속도 | 약간의 급격한 하강 구간 | 불규칙, 중간에 멈춤 가능 | 제어 불능, 급격 하강 |
| **상지 사용** | 불필요 | 간헐적 균형 보조 | 필수적 (의자 팔걸이, 바닥 짚기) | 필수적 + 성인 보조 |
| **착좌 충격** | 부드러운 착지 | 약간의 충격 | 다소 거친 착지 | 뚝 떨어지는 착지, 충격 큼 |

**Skeleton 기반 측정 가능 특성:**
- **Descent velocity profile:** pelvis z-축 속도의 시간 프로파일 → L1은 일정, L4는 급격한 가속
- **Controlled deceleration:** 착좌 직전 pelvis 감속률 → L1 높음 (제어됨), L4 낮음 (제어 불능)
- **Impact jerk at landing:** 착좌 순간 pelvis의 vertical jerk → L1 최소, L4 최대
- **Hand support detection:** wrist가 support surface에 접촉하는 시점/빈도 → L2부터 증가

#### 15.4.5 Side Rolling (옆구르기) 질적 차이 — L4 vs L5

| 질적 특성 | L4 | L5 |
|-----------|-----|-----|
| **자발성** | 자발적 개시 | 성인 보조로 개시하거나 자발적이나 매우 느림 |
| **대칭성** | 좌우 양 방향 가능 | 한 방향만 가능하거나 양 방향 모두 불완전 |
| **속도** | 비교적 빠르고 유연 | 매우 느림, 도중 멈춤 |
| **체간-하지 분리** | 체간과 하지가 분리된(segmental) 회전 | 체간과 하지가 일체(log-roll)로 회전 |
| **반복성** | 연속 2~3회 구르기 가능 | 1회 구르기 후 지치거나 재시도 어려움 |
| **상지 참여** | 상지로 추진력 보조, 자유로운 팔 운동 | 상지가 몸에 갇히거나 추진력 기여 미미 |

**Skeleton 기반 측정 가능 특성:**
- **Segmental rotation ratio:** shoulder roll onset과 hip roll onset의 시차 → L4는 분리 회전 (시차 > 0.3초), L5는 log-roll (시차 ≈ 0)
- **Rolling velocity:** 1회 완전 구르기 소요 시간 → L4 < L5
- **Bilateral symmetry:** 좌→우 vs 우→좌 구르기 속도 비율 → L4 ≈ 1.0, L5 편향
- **Inter-roll recovery time:** 연속 구르기 사이 정지 시간 → L4 < L5
- **Arm range of motion during roll:** shoulder abduction/flexion 범위 → L4 > L5

### 15.5 연령대별 Level 간 종합 구분점 (GMFCS-E&R Distinctions)

본 프로젝트 환자군(23~72개월)은 GMFCS-E&R의 3개 연령대에 걸쳐 있다. 연령대별로 동일 Level에서도 관찰되는 기능이 다르므로, 모델이 연령 정보를 활용할 수 있도록 연령대별 핵심 구분점을 정리한다.

#### 연령대 1: 2세 미만 (Before 2nd Birthday) — 본 데이터셋: lsa(L4, 23개월)

| 구분 | L1 | L2 | L3 | L4 | L5 |
|------|-----|-----|-----|-----|-----|
| **바닥 앉기** | 양손 자유로이 앉기 | 양손 사용하나 손으로 균형 보조 필요 시 있음 | 허리 지지 시에만 바닥 앉기 유지 | 머리 조절 있으나 앉기에 체간 지지 필요 | 항중력 자세 유지 불가 |
| **바닥 이동** | 네발기기 가능 | 배밀이 또는 네발기기 | 뒤집기와 배밀이 | 뒤집기만 가능 (앙와위→복와위) | 성인 보조 없이 뒤집기 불가 |
| **기립·보행** | 가구 잡고 서기 → 18~24개월에 독립 보행 | 가구 잡고 서기·걸음마 시도 가능 | — | — | — |

#### 연령대 2: 2~4세 (Between 2nd and 4th Birthday) — 본 데이터셋: kdu(L1,26mo), phm(L1,31mo), kto(L1,34mo), orj(L1,35mo), jeu(L2,33mo), kku(L3,28mo), oms(L5,29mo)

| 구분 | L1 | L2 | L3 | L4 | L5 |
|------|-----|-----|-----|-----|-----|
| **바닥 앉기** | 양손 자유 | 양손 자유하나 균형 어려움 | "W-sitting" 빈번, 성인 보조로 앉기 자세 취함 | 앉히면 유지하나 정렬·균형에 양손 지지 필요 | 항중력 자세 전반적 불가 |
| **이동 방법** | 보행이 선호 이동 수단 | 보조기구 사용한 보행이 선호 | 배밀이/네발기기 (비교대적), walker로 실내 단거리 | 뒤집기, 배밀이, 비교대 네발기기 (실내 단거리) | 독립 이동 수단 없음, 이송됨 |
| **보행** | 보조기구 없이 독립 보행 | 보조기구 선호 | walker + 성인 보조 (조향·회전) | — | — |

#### 연령대 3: 4~6세 (Between 4th and 6th Birthday) — 본 데이터셋의 다수 환자

| 구분 | L1 | L2 | L3 | L4 | L5 |
|------|-----|-----|-----|-----|-----|
| **의자 앉기→일어서기** | 손 지지 없이 일어남 | 안정된 표면에서 밀거나 당겨서 일어남 | 팔로 밀거나 당겨서 일어남 (안정 표면 필수) | 성인 보조 또는 안정 표면 필수 | 독립 불가, 적응 장비 필요 |
| **실내 보행** | 독립 보행 | 보조기구 없이 보행 | hand-held 보조기구로 보행 | walker + 성인 감독, 회전·불균형 표면 어려움 | 독립 이동 불가 |
| **실외 보행** | 독립 보행 + 계단 | 짧은 거리만, 장거리 시 바퀴 이동 | 자주 이송됨 | 이송, 동력 휠체어 가능 | 이송, 광범위 적응 장치 있는 동력 휠체어 일부 가능 |
| **달리기·점프** | 출현 | 불가 | — | — | — |
| **계단** | 독립 | 난간 잡기 필수 | 성인 보조 | 불가 | 불가 |

#### 15.5.1 모델 학습에 대한 시사점

1. **L1 vs L2 구분의 핵심은 "달리기·점프 능력"과 "계단 독립성"이다.** 그러나 본 데이터셋은 run/jump를 제외하였으므로, **보행 속도, cadence 규칙성, 좌우 대칭성, 장거리 보행 시 균형 유지** 같은 연속적 질적 지표로 대체해야 한다.

2. **L2 vs L3 구분은 가장 명확하다:** L2는 4세 이후 보조기구 없이 보행 가능, L3는 보조기구 필수. 3D skeleton에서 **상지가 보조기구를 파지하고 있는 패턴** (wrist 좌표가 hip 높이에서 고정됨)이 L3의 강력한 신호이다.

3. **L3 vs L4 구분의 핵심:**
   - L3: "안정된 표면에서 독립적으로 일어서고, 실내에서 walker로 보행"
   - L4: "성인 보조 없이는 일어서기 어려움, walker + 성인 감독, 회전·불균형 표면에서 실패"
   - → Skeleton 특성: seated_to_standing에서 **외부 지지점(사람/물체) 방향으로의 상지 뻗기 빈도**, walk에서 **방향전환 시 균형 상실 이벤트**

4. **연령 정보의 중요성:** 같은 L2라도 33개월(jeu)과 72개월(jty)의 운동 패턴은 다르다. 2~4세 L1은 "보행이 선호 이동수단"이 되는 시기이고, 4~6세 L1은 "달리기·점프 출현" 시기이다. **metadata_vector의 age_normalized가 이 차이를 반영하므로, 모델이 연령에 따라 동일 Level 내 기능 변동을 학습**할 수 있다.

5. **"할 수 없는 것" 자체가 분류 정보:** GMFCS-E&R은 각 Level에서 "할 수 없는 것"을 명시한다 (L2: 달리기·점프 불가 / L5: 독립 이동 수단 없음). 본 프로젝트의 metadata_vector (w_status, cr_status 등)가 이 정보를 직접 인코딩하며, **특정 동작 클립의 부재 자체가 Level 추론에 기여**한다.

---

## 16. 계층적 분류 전체 핵심 동작 종합표

### 16.1 분류 단계별 핵심 동작 배정

| 분류 단계 | 구분 목표 | 우선순위 | 핵심 동작 | 변별 포인트 |
|-----------|-----------|---------|-----------|------------|
| **Stage 1** | 보행 가능 vs 불가 | 1순위 | **walk (w)** | walk 유무 자체가 1차 분류 |
| | | 2순위 | **seated_to_standing (c_s)** | 전환 독립성으로 보조 판별 |
| **Stage 2-A** | **L1 vs L2 vs L3** | 1순위 | **walk (w)** | L1 빠르고 안정 → L2 약간 비대칭 → L3 느리고 불안정·보조기구 |
| | | 2순위 | **seated_to_standing (c_s)** | L1 즉각 독립 → L2 약간 지연 → L3 손짚기·보조 필요 |
| | | 3순위 | **standing_to_seated (s_c)** | 앉는 과정의 제어력 차이 |
| **L3 vs L4** 🔑 | **L3 vs L4 (핵심)** | 1순위 | **crawl (cr)** | L3 reciprocal crawl → L4 불완전·배밀이·비대칭 |
| | | 2순위 | **walk (w)** | L3 보조기구로 독립 보행 → L4 타인 보조 or 불가 |
| | | 3순위 | **seated_to_standing (c_s)** | L3 자력 기립(손짚기) → L4 타인 보조 필수 |
| **Stage 2-B** | **L4 vs L5** | 1순위 | **crawl (cr)** | L4 전원 가능 vs L5 대부분 불가 |
| | | 2순위 | **side_rolling (sr)** | 구르기 대칭성·속도·자발성의 질적 차이 |
| | | 3순위 | **seated_to_standing (c_s)** | 전환 능력의 질적 차이 |

### 16.2 동작별 역할 요약

| 동작 | 약어 | 관여하는 분류 단계 | 전체 역할 |
|------|------|-------------------|-----------|
| **walk** | w | Stage 1 + Stage 2-A + L3vsL4 | 🏆 **가장 다용도** — 보행 유무로 1차 분류, 보행 질로 L1/L2/L3 변별, L3/L4 독립성 변별 |
| **crawl** | cr | L3vsL4 + Stage 2-B | 🏆 **L3~L5 변별의 핵심** — 질적 차이(L3vsL4)와 가능 여부(L4vsL5) 모두 활용 |
| **seated_to_standing** | c_s | Stage 2-A + L3vsL4 + Stage 2-B | 🔧 **전 단계 보조** — 모든 분류에서 보조 변별력 제공 |
| **standing_to_seated** | s_c | Stage 2-A | 🔧 L1/L2/L3 보조 변별 |
| **side_rolling** | sr | Stage 2-B | 🎯 L4 vs L5 전용 변별 |

### 16.3 비디오 편집 시 동작별 우선순위

| 우선순위 | 동작 | 대상 환자 | 편집 목표 |
|---------|------|-----------|-----------|
| ⭐⭐⭐ | **walk (w)** | L1~L4 보행 가능자 (16명) | 모든 walk 구간 triplet 편집 |
| ⭐⭐⭐ | **crawl (cr)** | L3~L5 수행 가능자 | 모든 crawl 구간 triplet 편집 |
| ⭐⭐ | **seated_to_standing (c_s)** | L1~L5 수행 가능자 | 모든 전환 구간 triplet 편집 |
| ⭐⭐ | **standing_to_seated (s_c)** | L1~L4 수행 가능자 | 가능한 범위 내 triplet 편집 |
| ⭐ | **side_rolling (sr)** | L4~L5 (7명) | L4/L5 비보행 그룹 변별용 |

**참고:** L1·L2 환자의 crawl 데이터는 학습에 포함하지 않는다. 경증 환아에게 crawl은 "쉬워서 수행할 필요 없는" 동작이므로 동작 질 비교 대상이 아니다.

---

## 17. 동작 수행 체크리스트 및 메타데이터 벡터

### 17.1 환자별 5개 핵심 동작 수행 체크리스트

각 동작에 대해 3가지 상태를 부여한다:
- ✅ **수행함** (Performed): 해당 동작의 비디오 클립이 존재
- 🔵 **불필요** (Too Easy / Not Needed): 경증으로 해당 동작이 불필요 (L1·L2의 crawl/side_rolling 등)
- ❌ **수행불가** (Cannot Perform): 중증으로 해당 동작을 수행할 수 없음

| 환자 | GMFCS | 성별 | 월령 | w | cr | c_s | s_c | sr |
|------|-------|------|------|---|-----|-----|-----|-----|
| jyh | L1 | 남 | 58mo | ✅ 수행(23) | 🔵 불필요 | ✅ 수행(6) | ✅ 수행(2) | 🔵 불필요 |
| kdu | L1 | 남 | 26mo | ✅ 수행(80) | 🔵 불필요 | ✅ 수행(24) | ✅ 수행(22) | 🔵 불필요 |
| kra | L1 | 여 | 62mo | ✅ 수행(85) | 🔵 불필요 | ✅ 수행(3) | ⚠️ 데이터없음 | 🔵 불필요 |
| kto | L1 | 남 | 34mo | ✅ 수행(98) | 🔵 불필요 | ✅ 수행(33) | ✅ 수행(19) | 🔵 불필요 |
| orj | L1 | 여 | 35mo | ✅ 수행(14) | 🔵 불필요 | ✅ 수행(15) | ✅ 수행(15) | 🔵 불필요 |
| phm | L1 | 남 | 31mo | ✅ 수행(29) | 🔵 불필요 | ✅ 수행(14) | ✅ 수행(4) | 🔵 불필요 |
| hja | L2 | 여 | 50mo | ✅ 수행(35) | 🔵 불필요 | ✅ 수행(21) | ✅ 수행(21) | 🔵 불필요 |
| jeu | L2 | 여 | 33mo | ✅ 수행(45) | 🔵 불필요 | ✅ 수행(8) | ✅ 수행(2) | 🔵 불필요 |
| jji | L2 | 여 | 46mo | ✅ 수행(24) | 🔵 불필요 | ✅ 수행(21) | ✅ 수행(13) | 🔵 불필요 |
| jty | L2 | 남 | 72mo | ✅ 수행(27) | 🔵 불필요 | ✅ 수행(6) | ⚠️ 데이터없음 | 🔵 불필요 |
| lrl | L2 | 여 | 51mo | ✅ 수행(78) | 🔵 불필요 | ✅ 수행(35) | ✅ 수행(33) | 🔵 불필요 |
| kku | L3 | 남 | 28mo | ❌ 수행불가 | ✅ 수행(91) | ✅ 수행(13) | ✅ 수행(7) | 🔵 불필요 |
| ly | L3 | 남 | 38mo | ✅ 수행(11) | ✅ 수행(27) | ✅ 수행(15) | ✅ 수행(8) | 🔵 불필요 |
| mkj | L3 | 남 | 60mo | ✅ 수행(53) | ⚠️ 데이터없음 | ⚠️ 데이터없음 | ⚠️ 데이터없음 | 🔵 불필요 |
| pjw | L3 | 남 | 51mo | ✅ 수행(34) | ✅ 수행(2) | ✅ 수행(24) | ✅ 수행(15) | 🔵 불필요 |
| hdi | L4 | 여 | 62mo | ✅ 수행(25) | ✅ 수행(32) | ✅ 수행(15) | ✅ 수행(15) | ✅ 수행(6) |
| jrh | L4 | 남 | 52mo | ✅ 수행(21) | ✅ 수행(75) | ✅ 수행(39) | ✅ 수행(30) | ⚠️ 데이터없음 |
| lsa | L4 | 여 | 23mo | ❌ 수행불가 | ✅ 수행(9) | ❌ 수행불가 | ❌ 수행불가 | ✅ 수행(6) |
| ajy | L5 | 여 | 44mo | ❌ 수행불가 | ❌ 수행불가 | ❌ 수행불가 | ❌ 수행불가 | ✅ 수행(6) |
| hdu | L5 | 남 | 64mo | ❌ 수행불가 | ❌ 수행불가 | ❌ 수행불가 | ❌ 수행불가 | ✅ 수행(6) |
| kcw | L5 | 여 | 56mo | ❌ 수행불가 | ❌ 수행불가 | ❌ 수행불가 | ❌ 수행불가 | ✅ 수행(41) |
| kri | L5 | 남 | 62mo | ❌ 수행불가 | ✅ 수행(51) | ✅ 수행(3) | ❌ 수행불가 | ⚠️ 데이터없음 |
| oms | L5 | 남 | 29mo | ❌ 수행불가 | ❌ 수행불가 | ✅ 수행(3) | ❌ 수행불가 | ✅ 수행(36) |
| pjo | L5 | 남 | 65mo | ❌ 수행불가 | ✅ 수행(12) | ✅ 수행(6) | ✅ 수행(6) | ✅ 수행(27) |

### 17.2 GMFCS Level별 동작 수행 패턴 시그니처

| GMFCS | w | cr | c_s | s_c | sr | 패턴 특징 |
|-------|---|-----|-----|-----|-----|-----------|
| **L1** | ✅ | 🔵 | ✅ | ✅ | 🔵 | 전부 가능, crawl/rolling 불필요 |
| **L2** | ✅ | 🔵 | ✅ | ✅ | 🔵 | L1과 동일 패턴 → **영상 질로만 구분** |
| **L3** | ✅/❌ | ✅ | ✅ | ✅ | 🔵 | walk 불가 시작, crawl 주력화 |
| **L4** | ✅/❌ | ✅ | ✅/❌ | ✅/❌ | ✅ | 전반적 기능 저하, rolling 필요 |
| **L5** | ❌ | ❌/✅ 소수 | ❌/✅ 극소 | ❌ | ✅ | 대부분 불가, rolling만 가능 |

**핵심 발견:** L1과 L2의 동작 수행 패턴이 완전히 동일하여 메타데이터만으로는 구분이 불가하다. 반면 L3 이하는 "무엇을 할 수 있는가" 자체가 강력한 분류 정보를 제공한다.

### 17.3 메타데이터 벡터 설계

동작 수행 체크리스트를 모델 입력의 보조 정보로 활용하기 위해, 환자 단위 메타데이터 벡터를 정의한다. 이 벡터는 **training, validation, inference 전 단계에서** 3D skeleton 데이터와 함께 모델에 제공된다.

```
metadata_vector = [
    sex,              # 0=여, 1=남
    age_normalized,   # 월령 / 72 (0~1 정규화)
    w_status,         # 0=수행불가, 1=수행, 2=불필요   (walk)
    cr_status,        # 0=수행불가, 1=수행, 2=불필요   (crawl)
    c_s_status,       # 0=수행불가, 1=수행, 2=불필요   (seated_to_standing)
    s_c_status,       # 0=수행불가, 1=수행, 2=불필요   (standing_to_seated)
    sr_status,        # 0=수행불가, 1=수행, 2=불필요   (side_rolling)
]
# → 7차원 벡터, 모든 클립에 환자 단위로 동일하게 부여
```

**인코딩 규칙:**

| 상태 | 코드 | 의미 |
|------|------|------|
| 0 (수행불가) | ❌ | 중증으로 해당 동작을 수행할 수 없음 |
| 1 (수행) | ✅ | 해당 동작의 비디오 클립이 존재하며 학습 데이터로 사용 |
| 2 (불필요) | 🔵 | 경증으로 해당 동작이 불필요 (할 수 있지만 임상적 의미 없음) |

**Inference 단계 적용:** 새 환자 촬영 전 보호자/치료사가 간단한 체크리스트를 작성한다:
- "이 아이가 걸을 수 있나요?" → w_status
- "기기(배밀이/네발기기)가 가능한가요?" → cr_status
- "바닥에서 혼자 일어설 수 있나요?" → c_s_status
- "서서 혼자 앉을 수 있나요?" → s_c_status
- "옆구르기가 가능한가요?" → sr_status

이 방식은 추가 촬영 없이 얻을 수 있는 정보이므로 **실용성이 매우 높으며**, 임상 현장에서 GMFCS 사전 평가 절차와 자연스럽게 통합된다 [10].

---

## 18. 참고 문헌

[1] P. Zhao, M. Alencastre-Miranda, Z. Shen, C. O'Neill, D. Whiteman, J. Gervas-Arruga, and H. I. Krebs, "Computer Vision for Gait Assessment in Cerebral Palsy: Metric Learning and Confidence Estimation," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 32, pp. 2336–2345, 2024. DOI: 10.1109/TNSRE.2024.3416159
→ https://ieeexplore.ieee.org/document/10560023/

[2] Skarimva, "Skeleton-based Action Recognition is a Multi-view Application," *ICASSP 2025 / arXiv:2602.23231*, 2025.
→ https://arxiv.org/html/2602.23231v1

[3] "Enhancing Cerebral Palsy Gait Analysis with 3D Computer Vision: A Dual-Camera Approach," *10th International Conference on Control, Decision and Information Technologies (CoDIT)*, IEEE, 2024.
→ https://ieeexplore.ieee.org/document/10708137/

[4] K. Liu, L. Chen, L. Xie, J. Yin, S. Gan, Y. Yan, and E. Yin, "Auto calibration of multi-camera system for human pose estimation," *IET Computer Vision*, vol. 16, no. 8, pp. 660–673, 2022. DOI: 10.1049/cvi2.12130
→ https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.12130

[5] K. Takahashi, D. Mikami, M. Isogawa, and H. Kimata, "Human Pose as Calibration Pattern; 3D Human Pose Estimation with Multiple Unsynchronized and Uncalibrated Cameras," *IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2018.
→ https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w34/Takahashi_Human_Pose_As_CVPR_2018_paper.pdf

[6] M. Pätzold, S. Bultmann, and S. Behnke, "Online Marker-Free Extrinsic Camera Calibration Using Person Keypoint Detections," *DAGM German Conference on Pattern Recognition (GCPR)*, Lecture Notes in Computer Science, vol. 13485, Springer, 2022.
→ https://arxiv.org/abs/2209.07393

[7] S. Liu, D. Y. Yao, S. Gupta, and S. Wang, "VisualSync: Multi-Camera Synchronization via Cross-View Object Motion," *NeurIPS 2025 / arXiv:2512.02017*, 2025.
→ https://arxiv.org/abs/2512.02017

[8] R. Kavi, V. Kulathumani, F. Rohit, and V. Kecojevic, "Multiview fusion for activity recognition using deep neural networks," *Journal of Electronic Imaging*, vol. 25, no. 4, 043010, 2016. DOI: 10.1117/1.JEI.25.4.043010
→ https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-25/issue-4/043010/Multiview-fusion-for-activity-recognition-using-deep-neural-networks/10.1117/1.JEI.25.4.043010.short

[9] A. Shahroudy, J. Liu, T.-T. Ng, and G. Wang, "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis," *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016.
→ https://github.com/shahroudy/NTURGB-D

[10] R. Palisano, P. Rosenbaum, D. Bartlett, and M. Livingston, "GMFCS – E&R: Gross Motor Function Classification System – Expanded and Revised," CanChild Centre for Childhood Disability Research, McMaster University, 2007.
→ https://canchild.ca/resources/42-gmfcs-e-r/

[11] "Multiview child motor development dataset for AI-driven assessment of child development," *GigaScience*, vol. 12, giad039, Oxford Academic, 2023. DOI: 10.1093/gigascience/giad039
→ https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giad039/7181060

[12] SteerPose: "Simultaneous Extrinsic Camera Calibration and Matching from Articulation," *arXiv:2506.01691*, 2025.
→ https://arxiv.org/abs/2506.01691

---

*Generated: 2026-03-20 | Updated: 2026-03-31 (Sections 15.4~15.5 추가: GMFCS-E&R 기반 동작 질적 차이 기술서, 연령대별 Level 간 종합 구분점, Skeleton 기반 측정 가능 특성 매핑)*
