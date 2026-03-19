import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ETRI 소액 입찰 산정 대시보드",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS 커스텀 스타일
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* 전체 배경 – 흰색 */
  .stApp { background-color: #f5f7fa; }

  /* 최적 투찰 카드 (파란 계열) */
  .optimal-card {
      background: linear-gradient(135deg, #e8f4fd 0%, #d0eaf9 100%);
      border: 2px solid #1a73e8;
      border-radius: 16px;
      padding: 28px 32px;
      text-align: center;
      box-shadow: 0 4px 20px rgba(26,115,232,0.18);
      margin: 8px 0;
  }
  .optimal-card .label {
      color: #1a73e8;
      font-size: 0.95rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
  }
  .optimal-card .amount {
      color: #0d47a1;
      font-size: 2.6rem;
      font-weight: 800;
      letter-spacing: 0.02em;
  }
  .optimal-card .sub {
      color: #5a7fa8;
      font-size: 0.88rem;
      margin-top: 8px;
  }

  /* 최고 투찰 카드 (주황 계열) */
  .max-card {
      background: linear-gradient(135deg, #fff8e8 0%, #fdecc8 100%);
      border: 2px solid #f59e0b;
      border-radius: 16px;
      padding: 28px 32px;
      text-align: center;
      box-shadow: 0 4px 20px rgba(245,158,11,0.18);
      margin: 8px 0;
  }
  .max-card .label {
      color: #b45309;
      font-size: 0.95rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
  }
  .max-card .amount {
      color: #78350f;
      font-size: 2.6rem;
      font-weight: 800;
      letter-spacing: 0.02em;
  }
  .max-card .sub {
      color: #92714a;
      font-size: 0.88rem;
      margin-top: 8px;
  }

  /* 지표 카드 */
  .metric-card {
      background: #ffffff;
      border-radius: 12px;
      padding: 18px 20px;
      text-align: center;
      border: 1px solid #e0e7ef;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .metric-card .m-label {
      color: #6b7a8d;
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
  }
  .metric-card .m-value {
      color: #1e293b;
      font-size: 1.3rem;
      font-weight: 700;
      margin-top: 6px;
  }

  /* 섹션 구분선 */
  hr { border-color: #dde3ec; }

  /* 사이드바 */
  section[data-testid="stSidebar"] {
      background-color: #ffffff;
      border-right: 1px solid #e0e7ef;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────
def fmt_krw(value: float) -> str:
    """정수 원화 포맷 (예: 1,234,567,890 원)"""
    return f"{int(value):,} 원"


def generate_preliminary_prices(base: float, n: int = 15, seed: int | None = None) -> np.ndarray:
    """기초금액의 ±2% 범위 내에서 n개 복수예비가격 생성."""
    rng = np.random.default_rng(seed)
    low  = base * 0.98
    high = base * 1.02
    prices = rng.uniform(low, high, size=n)
    return np.sort(prices)


def get_mode_bin(data: np.ndarray, base: float, n_bins: int = 60) -> dict:
    """히스토그램 최빈 구간(mode bin) 및 사정률 분석."""
    counts, edges = np.histogram(data, bins=n_bins)
    peak_idx = int(np.argmax(counts))
    low, high = edges[peak_idx], edges[peak_idx + 1]
    return {
        "low":        low,
        "high":       high,
        "count":      int(counts[peak_idx]),
        "ratio":      counts[peak_idx] / len(data),       # 전체 대비 비율
        "rate_low":   (low  / base) * 100,                # 사정률 하한 (%)
        "rate_high":  (high / base) * 100,                # 사정률 상한 (%)
        "mid":        (low + high) / 2,                   # 구간 중앙값
        "rate_mid":   ((low + high) / 2 / base) * 100,   # 중앙 사정률 (%)
    }


def run_simulation(prelim_prices: np.ndarray, n_select: int, n_sim: int) -> np.ndarray:
    """
    매 시뮬레이션마다 prelim_prices 에서 n_select 개를 무작위 추출해
    산술 평균(예정가격)을 계산하고, n_sim 개의 결과를 반환.
    """
    rng = np.random.default_rng(42)
    n = len(prelim_prices)
    results = np.empty(n_sim)
    for i in range(n_sim):
        chosen = rng.choice(prelim_prices, size=n_select, replace=False)
        results[i] = chosen.mean()
    return results


# ─────────────────────────────────────────────
# 사이드바 – 입력 파라미터
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 파라미터 설정")
    st.divider()

    base_input = st.number_input(
        "📌 기초금액 (원)",
        min_value=1_000_000,
        max_value=10_000_000_000,
        value=1_000_000_000,
        step=1_000_000,
        format="%d",
        help="공고상 명시된 기초금액을 입력하세요.",
    )

    st.markdown("---")

    lower_rate_pct = st.slider(
        "📉 낙찰하한율 (%)",
        min_value=80.0,
        max_value=95.0,
        value=88.0,
        step=0.1,
        format="%.1f%%",
        help="ETRI 소액계약 기본값 88%",
    )
    lower_rate = lower_rate_pct / 100

    n_sim = st.select_slider(
        "🔁 시뮬레이션 횟수",
        options=[1_000, 3_000, 5_000, 10_000, 30_000, 50_000],
        value=10_000,
    )

    n_select = st.slider(
        "🎲 추첨 선택 개수",
        min_value=2,
        max_value=6,
        value=4,
        help="전체 입찰자 투표 후 최다득표 순 4개 선정 (기본값)",
    )

    st.markdown("---")
    run_btn = st.button("▶  시뮬레이션 실행", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("ETRI 소액입찰 낙찰자 결정 방식 모사\n\nPRD v1.0 · Streamlit 대시보드")


# ─────────────────────────────────────────────
# 메인 헤더
# ─────────────────────────────────────────────
st.markdown("# 🏗️ ETRI 소액 입찰 금액 산정 대시보드")
st.markdown(
    "기초금액 ±2% 범위 내 **15개 복수예비가격**을 생성하고, "
    f"**{n_sim:,}회 몬테카를로 시뮬레이션**으로 예정가격 분포를 분석합니다."
)
st.divider()


# ─────────────────────────────────────────────
# 세션 상태 초기화
# ─────────────────────────────────────────────
if "prelim_prices" not in st.session_state:
    st.session_state.prelim_prices = None
    st.session_state.sim_results   = None
    st.session_state.base          = None


# ─────────────────────────────────────────────
# 실행 버튼 → 계산
# ─────────────────────────────────────────────
if run_btn:
    with st.spinner("시뮬레이션 실행 중…"):
        prelim = generate_preliminary_prices(base_input, n=15)
        sims   = run_simulation(prelim, n_select=n_select, n_sim=n_sim)
        st.session_state.prelim_prices = prelim
        st.session_state.sim_results   = sims
        st.session_state.base          = base_input


# ─────────────────────────────────────────────
# 결과 표시
# ─────────────────────────────────────────────
if st.session_state.sim_results is not None:
    prelim  = st.session_state.prelim_prices
    sims    = st.session_state.sim_results
    base    = st.session_state.base

    mean_p  = sims.mean()
    std_p   = sims.std()
    min_p   = sims.min()
    max_p   = sims.max()
    optimal = mean_p * lower_rate   # 최적 투찰 (하한)
    max_bid = mean_p                # 최고 투찰 (예정가격 = 상한, 초과 시 무효)
    mode_bin = get_mode_bin(sims, base)

    # ── 투찰 금액 강조 카드 2개 ──────────────────
    card_l, card_r = st.columns(2)
    with card_l:
        st.markdown(f"""
        <div class="optimal-card">
          <div class="label">✅ 최적 추천 투찰 금액 &nbsp;(낙찰하한율 {lower_rate_pct:.1f}%)</div>
          <div class="amount">{int(optimal):,} 원</div>
          <div class="sub">
            평균 예정가격 {int(mean_p):,} 원 × {lower_rate_pct:.1f}%<br>
            이 금액 이상으로 투찰해야 낙찰 조건 충족
          </div>
        </div>
        """, unsafe_allow_html=True)
    with card_r:
        st.markdown(f"""
        <div class="max-card">
          <div class="label">⚠️ 최고 투찰 금액 &nbsp;(예정가격 상한 · 초과 시 무효)</div>
          <div class="amount">{int(max_bid):,} 원</div>
          <div class="sub">
            평균 예정가격 {int(mean_p):,} 원 × 100%<br>
            투찰 유효 범위: {int(optimal):,} ~ {int(max_bid):,} 원
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("&nbsp;")  # 여백

    # ── 지표 카드 4개 ────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("📊 평균 예정가격",  mean_p),
        ("📈 최고 예정가격",  max_p),
        ("📉 최저 예정가격",  min_p),
        ("〰️ 표준편차",       std_p),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="m-label">{label}</div>
              <div class="m-value">{int(val):,} 원</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── 히스토그램 + 세부 통계 ──────────────────
    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        st.subheader("📊 예정가격 분포 (히스토그램)")

        fig = go.Figure()

        # 히스토그램 본체
        fig.add_trace(go.Histogram(
            x=sims,
            nbinsx=60,
            marker_color="#3a86ff",
            opacity=0.85,
            name="예정가격 빈도",
        ))

        # 평균 수직선
        fig.add_vline(
            x=mean_p,
            line=dict(color="#1a73e8", width=2, dash="dash"),
            annotation_text=f"평균<br>{int(mean_p):,}",
            annotation_position="top right",
            annotation_font_color="#1a73e8",
        )

        # 최고 투찰 상한선
        fig.add_vline(
            x=max_bid,
            line=dict(color="#f59e0b", width=2, dash="dash"),
            annotation_text=f"최고 투찰<br>{int(max_bid):,}",
            annotation_position="top right",
            annotation_font_color="#b45309",
        )

        # 낙찰하한선
        fig.add_vline(
            x=optimal,
            line=dict(color="#e11d48", width=2.5),
            annotation_text=f"투찰 하한<br>{int(optimal):,}",
            annotation_position="top left",
            annotation_font_color="#e11d48",
        )

        # 기초금액선
        fig.add_vline(
            x=base,
            line=dict(color="#64b5f6", width=1.5, dash="dot"),
            annotation_text=f"기초금액<br>{int(base):,}",
            annotation_position="top right",
            annotation_font_color="#1976d2",
        )

        # ±1σ 음영
        fig.add_vrect(
            x0=mean_p - std_p, x1=mean_p + std_p,
            fillcolor="rgba(245,158,11,0.07)",
            line_width=0,
            annotation_text="±1σ",
            annotation_position="top left",
            annotation_font_color="#999",
        )

        # 유효 투찰 구간 음영 (하한~상한)
        fig.add_vrect(
            x0=optimal, x1=max_bid,
            fillcolor="rgba(26,115,232,0.06)",
            line_width=0,
        )

        # 최빈 구간 강조 (진한 초록 음영 + 어노테이션)
        fig.add_vrect(
            x0=mode_bin["low"], x1=mode_bin["high"],
            fillcolor="rgba(16,185,129,0.18)",
            line=dict(color="#10b981", width=1.5, dash="dot"),
            annotation_text=(
                f"최빈 구간<br>"
                f"사정률 {mode_bin['rate_low']:.3f}~{mode_bin['rate_high']:.3f}%<br>"
                f"({mode_bin['ratio']*100:.1f}%)"
            ),
            annotation_position="top right",
            annotation_font_color="#059669",
            annotation_font_size=11,
        )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f9fbfd",
            margin=dict(l=10, r=10, t=30, b=40),
            xaxis_title="예정가격 (원)",
            yaxis_title="빈도",
            showlegend=False,
            height=420,
            xaxis=dict(tickformat=",", gridcolor="#e8edf3"),
            yaxis=dict(gridcolor="#e8edf3"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.subheader("🎰 15개 복수예비가격")
        df_prelim = pd.DataFrame({
            "번호": range(1, 16),
            "예비가격 (원)": [f"{int(p):,}" for p in prelim],
            "사정률 (%)": [f"{(p/base)*100:.4f}%" for p in prelim],
        })
        st.dataframe(
            df_prelim,
            hide_index=True,
            use_container_width=True,
            height=480,
        )

    st.divider()

    # ── 수렴 과정 라인 차트 ──────────────────────
    st.subheader("📈 시뮬레이션 수렴 과정 (누적 평균)")

    step = max(1, n_sim // 500)          # 최대 500 포인트만 플롯
    idx  = np.arange(step, n_sim + 1, step)
    cumulative_mean = np.array([sims[:i].mean() for i in idx])
    cumulative_std  = np.array([sims[:i].std()  for i in idx])

    fig2 = go.Figure()

    # ±σ 음영대
    fig2.add_trace(go.Scatter(
        x=np.concatenate([idx, idx[::-1]]),
        y=np.concatenate([
            cumulative_mean + cumulative_std,
            (cumulative_mean - cumulative_std)[::-1],
        ]),
        fill="toself",
        fillcolor="rgba(58,134,255,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1σ 범위",
        showlegend=True,
    ))

    # 누적 평균선
    fig2.add_trace(go.Scatter(
        x=idx, y=cumulative_mean,
        mode="lines",
        line=dict(color="#3a86ff", width=2),
        name="누적 평균 예정가격",
    ))

    # 최종 평균 기준선
    fig2.add_hline(
        y=mean_p,
        line=dict(color="#1a73e8", width=1.5, dash="dash"),
        annotation_text=f"수렴값 {int(mean_p):,} 원",
        annotation_font_color="#1a73e8",
    )

    fig2.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f9fbfd",
        margin=dict(l=10, r=10, t=20, b=40),
        xaxis_title="시뮬레이션 횟수",
        yaxis_title="누적 평균 예정가격 (원)",
        height=350,
        yaxis=dict(tickformat=",", gridcolor="#e8edf3"),
        xaxis=dict(gridcolor="#e8edf3"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── 요약 해석 ────────────────────────────────
    st.divider()
    st.subheader("📋 분석 요약")
    pct_rate = (mean_p / base) * 100
    col_a, col_b = st.columns(2)
    with col_a:
        st.info(f"""
**예정가격 분석**
- 기초금액 대비 평균 사정률: **{pct_rate:.4f}%**
- 표준편차: {int(std_p):,} 원 (변동계수 {(std_p/mean_p*100):.3f}%)
- 분포 구간: {int(min_p):,} ~ {int(max_p):,} 원
        """)
        st.success(f"""
**최빈 구간 사정률 분석 (F-102)**
- 최빈 구간: **{int(mode_bin['low']):,} ~ {int(mode_bin['high']):,} 원**
- 기초금액 대비 사정률: **{mode_bin['rate_low']:.3f}% ~ {mode_bin['rate_high']:.3f}%**
- 구간 중앙 사정률: **{mode_bin['rate_mid']:.4f}%**
- 전체 시뮬레이션 중 해당 구간 빈도: **{mode_bin['count']:,}회 ({mode_bin['ratio']*100:.1f}%)**
        """)
    with col_b:
        st.warning(f"""
**투찰 전략 권고**
- ✅ 최적 투찰액: **{int(optimal):,} 원** (하한)
- ⚠️ 최고 투찰액: **{int(max_bid):,} 원** (상한 · 초과 시 무효)
- 유효 투찰 범위: {int(optimal):,} ~ {int(max_bid):,} 원
- 경쟁률에 따라 하한 대비 ±{int(std_p*0.5):,} 원 범위 조정 검토
        """)

else:
    # 실행 전 안내 화면
    st.markdown("""
    <div style="text-align:center; padding: 80px 0; color: #94a3b8;">
      <div style="font-size: 4rem;">🏗️</div>
      <div style="font-size: 1.4rem; font-weight: 600; margin-top: 16px; color: #1a73e8;">
        사이드바에서 기초금액을 입력하고<br>시뮬레이션을 실행하세요
      </div>
      <div style="margin-top: 12px; font-size: 0.95rem; color: #64748b;">
        ±2% 범위 내 15개 복수예비가격 생성 →
        10,000회 몬테카를로 시뮬레이션 →
        최적 · 최고 투찰금액 산정
      </div>
    </div>
    """, unsafe_allow_html=True)
