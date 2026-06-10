import streamlit as st
import pandas as pd  # type: ignore[import-untyped]
import time
from lc_shift import (
    RouterConfig,
    RouterShifter,
    ShiftRequest,
    Strategy,
    PRESETS,
    RoutingDecision,
)
from lc_shift.strategies import compute_complexity, estimate_token_count

# Page Setup
st.set_page_config(
    page_title="lc-shift Playground",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for rich aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #0f1116;
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #a0aec0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1a202c;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# App Title & Subtitle
st.markdown('<div class="main-header">⚡ lc-shift Playground</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Zero-overhead, local LLM prompt routing & budget control gateway</div>', unsafe_allow_html=True)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "spent_usd" not in st.session_state:
    st.session_state.spent_usd = 0.0
if "requests_count" not in st.session_state:
    st.session_state.requests_count = 0
if "cache_hits" not in st.session_state:
    st.session_state.cache_hits = 0

# Sidebar Configuration
st.sidebar.title("⚙️ Router Configuration")

preset_name = st.sidebar.selectbox(
    "Choose Preset Tiers",
    options=list(PRESETS.keys()),
    index=2  # default to mixed-frontier
)
selected_tiers = PRESETS[preset_name]

strategy_opt = st.sidebar.selectbox(
    "Routing Strategy",
    options=[s.value for s in Strategy],
    index=0  # complexity
)

# Render Strategy Parameters
cost_budget = None
latency_target = None
complexity_threshold = 0.5
semantic_routes = None
classifier_weights = None
classifier_intercept = 0.0
classifier_threshold = 0.5

if strategy_opt == "cost_aware":
    cost_budget = st.sidebar.number_input("Cost Budget (USD)", min_value=0.01, max_value=10.0, value=1.00, step=0.1)
elif strategy_opt == "latency":
    latency_target = st.sidebar.slider("Latency Target (ms)", min_value=100, max_value=3000, value=1200, step=50)
elif strategy_opt == "complexity":
    complexity_threshold = st.sidebar.slider("Complexity Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
elif strategy_opt == "semantic":
    st.sidebar.markdown("### Semantic Utterances")
    performance_utts = st.sidebar.text_area(
        "Performance Tier Prompts",
        value="write code, recursive algorithm, compile rust, math derivation, prove theory"
    )
    economy_utts = st.sidebar.text_area(
        "Economy Tier Prompts",
        value="hi, hello, say thanks, weather today, tell me a joke, greeting"
    )
    
    # Process text inputs into route mappings
    semantic_routes = {
        "performance": [u.strip() for u in performance_utts.split(",") if u.strip()],
        "economy": [u.strip() for u in economy_utts.split(",") if u.strip()]
    }
elif strategy_opt == "classifier":
    st.sidebar.markdown("### Classifier Weights")
    classifier_intercept = st.sidebar.number_input("Intercept (Bias)", value=-0.5, step=0.1)
    classifier_threshold = st.sidebar.slider("Classification Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    # Weights configuration in CSV-like format
    weights_raw = st.sidebar.text_area(
        "Weights (token: weight)",
        value="code: 2.5\nalgorithm: 2.0\nmathematics: 1.8\nhi: -2.0\nhello: -2.0\nsimple: -1.5"
    )
    classifier_weights = {}
    for line in weights_raw.split("\n"):
        if ":" in line:
            token, val = line.split(":")
            try:
                classifier_weights[token.strip().lower()] = float(val.strip())
            except ValueError:
                pass

# Build RouterConfig
default_tier = "performance"
if "balanced" in selected_tiers:
    default_tier = "balanced"
elif "economy" in selected_tiers:
    default_tier = "economy"
else:
    default_tier = list(selected_tiers.keys())[0]

router_config = RouterConfig(
    tiers=selected_tiers,
    default_tier=default_tier,
    strategy=Strategy(strategy_opt),
    cost_budget_usd=cost_budget,
    latency_target_ms=latency_target,
    complexity_threshold=complexity_threshold,
    semantic_routes=semantic_routes,
    classifier_weights=classifier_weights,
    classifier_intercept=classifier_intercept,
    classifier_threshold=classifier_threshold
)

# Instantiate RouterShifter. The router is recreated on every rerun so config
# changes always take effect; cumulative spend is restored from session state.
router = RouterShifter(router_config)
router.seed_spend(st.session_state.spent_usd)

# Layout: Two Columns
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📝 Sandbox Input")
    
    sample_prompts = [
        "What is 2 + 2?",
        "Write a Python function to compute the Fibonacci sequence using memoization.",
        "Could you explain the difference between REST and gRPC and discuss performance trade-offs?",
        "Hi, how's the weather today?",
        "Analyze the security implications of utilizing JWTs for session storage in single-page apps.",
        "Tell me a short joke."
    ]
    
    selected_sample = st.selectbox("Or choose a sample prompt:", sample_prompts)
    
    user_prompt = st.text_area(
        "Prompt to Route",
        value=selected_sample,
        height=150
    )
    
    sim_token_in = st.number_input("Simulate Input Tokens", min_value=1, value=150)
    sim_token_out = st.number_input("Simulate Output Tokens", min_value=1, value=250)
    
    run_btn = st.button("Route Prompt ⚡")

with col2:
    st.subheader("📊 Router Metrics")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric("Requests Routed", st.session_state.requests_count)
        st.metric("Total Cost (USD)", f"${st.session_state.spent_usd:.5f}")
    with m_col2:
        budget_str = f"${cost_budget:.2f}" if cost_budget else "Infinite"
        st.metric("Total Budget Limit", budget_str)
        rem_pct = "100%"
        if cost_budget:
            rem_pct = f"{max(0.0, (cost_budget - st.session_state.spent_usd) / cost_budget * 100):.1f}%"
        st.metric("Budget Remaining", rem_pct)

    if st.button("Reset Session Stats 🔄"):
        st.session_state.history = []
        st.session_state.spent_usd = 0.0
        st.session_state.requests_count = 0
        st.session_state.cache_hits = 0
        st.rerun()

# Execution and Results Display
if run_btn and user_prompt:
    # Run Routing
    req = ShiftRequest(prompt=user_prompt)
    
    # Calculate execution time
    t0 = time.perf_counter_ns()
    # Check exact cache simulation
    cached = False
    
    # Simple cached lookup sim matching actual RoutingCache
    if any(h["prompt"] == user_prompt and h["strategy"] == strategy_opt for h in st.session_state.history):
        cached = True
        st.session_state.cache_hits += 1
        
    async def run_routing() -> RoutingDecision:
        return await router.route(req)
        
    import asyncio
    decision = asyncio.run(run_routing())
    overhead_ms = (time.perf_counter_ns() - t0) / 1_000_000
    
    # Record simulated usage
    router.record_usage(decision.tier_name, input_tokens=sim_token_in, output_tokens=sim_token_out)
    st.session_state.spent_usd = router.spent_usd
    st.session_state.requests_count += 1
    
    # Save History
    st.session_state.history.append({
        "prompt": user_prompt,
        "strategy": strategy_opt,
        "tier": decision.tier_name,
        "reason": decision.reason,
        "overhead_ms": overhead_ms if not cached else 0.0,
        "cached": cached,
        "cost": (sim_token_in/1000) * decision.tier.cost_per_1k_input + (sim_token_out/1000) * decision.tier.cost_per_1k_output
    })
    
    # Display routing result banner
    st.markdown("---")
    st.subheader("🎯 Routing Decision")
    
    banner_color = "#3182ce"
    if "performance" in decision.tier_name or "smart" in decision.tier_name:
        banner_color = "#e53e3e"
    elif "economy" in decision.tier_name or "local" in decision.tier_name:
        banner_color = "#38a169"
        
    st.markdown(f"""
    <div style="background: {banner_color}; color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">
        <h3 style="margin: 0; color: white;">Selected Tier: {decision.tier_name.upper()}</h3>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;"><strong>Model</strong>: {decision.tier.name} ({decision.tier.provider}/{decision.tier.model_id})</p>
        <p style="margin: 0.2rem 0 0 0; font-size: 0.95rem;"><strong>Decision Overhead</strong>: {overhead_ms:.4f} ms | <strong>Cache Hit</strong>: {cached}</p>
        <p style="margin: 0.5rem 0 0 0; font-style: italic; background: rgba(0,0,0,0.2); padding: 0.5rem; border-radius: 5px;">Reason: {decision.reason}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual Breakdown of prompt analysis
    with st.expander("🔍 Interactive Prompt Heuristics Analyzer"):
        tokens = estimate_token_count(user_prompt)
        comp_score = compute_complexity(user_prompt)
        
        c_col1, c_col2, c_col3 = st.columns(3)
        c_col1.metric("Estimated Tokens", tokens)
        c_col2.metric("Code Block Detected", "Yes" if "```" in user_prompt or "`" in user_prompt else "No")
        c_col3.metric("Complexity Heuristic Score", f"{comp_score:.2f}")
        
        st.progress(comp_score)
        
        # Explain heuristic scoring parameters
        st.markdown("""
        **Heuristic Weights Calculation:**
        * Token length > 50: `+0.1`, > 200: `+0.2`, > 500: `+0.3`
        * Contains markdown code blocks: `+0.25`
        * Contains reasoning keywords (e.g. compare, evaluate, explain, analyze): `+0.05` per word (capped at `0.25`)
        * Contains multi-part list sequences (e.g. 1., first, additionally): `+0.05` per match (capped at `0.2`)
        """)

# Configured Tiers Metadata
st.markdown("---")
st.subheader("📚 Configured Tiers & Pricing")
tiers_data = []
for name, tier in selected_tiers.items():
    tiers_data.append({
        "Tier": name,
        "Model Name": tier.name,
        "Provider": tier.provider,
        "Model ID": tier.model_id,
        "Input Cost / 1k": f"${tier.cost_per_1k_input:.6f}",
        "Output Cost / 1k": f"${tier.cost_per_1k_output:.6f}",
        "Avg Latency (ms)": f"{tier.avg_latency_ms} ms"
    })
st.table(pd.DataFrame(tiers_data))

# History log
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Routing History Log")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df.style.format({
        "overhead_ms": "{:.4f} ms",
        "cost": "${:.6f}"
    }))
