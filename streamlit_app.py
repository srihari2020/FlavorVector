import plotly.graph_objects as go
import streamlit as st
from app.engine import FlavorEngine

st.set_page_config(
    page_title="FlavorVector | Real-Time Food-Science Search",
    page_icon="🧪",
    layout="wide",
)

TASTE_AXES = [
    ("sweet", "Sweet"),
    ("salty", "Salty"),
    ("sour", "Sour"),
    ("bitter", "Bitter"),
    ("umami", "Umami"),
    ("spicy_pungent", "Spicy / Pungent"),
    ("astringent", "Astringent"),
    ("cooling", "Cooling"),
    ("fatty_rich", "Fatty / Rich"),
    ("kokumi", "Kokumi"),
]

TASTE_OPTIONS = ["All Tastes"] + [key for key, _ in TASTE_AXES]

CUISINE_OPTIONS = [
    "All Corners",
    "Indian & South Asian",
    "East & Southeast Asian",
    "Mediterranean & European",
    "Latin American",
    "Middle Eastern & North African",
    "Baking & Pastry",
    "Universal Pantry",
    "Live Web Atlas",
]


def _taste_vector(tastes_csv: str, score_boost: float = 0.85) -> list[float]:
    active = {t.strip().lower() for t in tastes_csv.split(",") if t.strip()}
    return [round(score_boost if key in active else 0.15, 2) for key, _ in TASTE_AXES]


def render_substitution_radar(hit: dict) -> go.Figure:
    labels = [label for _, label in TASTE_AXES]
    sub_vals = _taste_vector(hit.get("tastes", ""), score_boost=max(0.65, hit["relevance_score"]))
    target_vals = _taste_vector(hit.get("tastes", ""), score_boost=0.90)

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=target_vals + [target_vals[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Target Role",
            line=dict(color="#38bdf8", width=2),
            fillcolor="rgba(56, 189, 248, 0.15)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=sub_vals + [sub_vals[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Substitute Match",
            line=dict(color="#f43f5e", width=2),
            fillcolor="rgba(244, 63, 94, 0.22)",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(tickfont=dict(size=10, color="#cbd5e1")),
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0.1, font=dict(size=10)),
        margin=dict(l=35, r=35, t=20, b=35),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_pairing_radar(suggestions: list[dict]) -> go.Figure:
    labels = [label for _, label in TASTE_AXES]
    agg = [0.15] * len(TASTE_AXES)
    for item in suggestions[:5]:
        vec = _taste_vector(item.get("tastes", ""), score_boost=item["relevance_score"])
        agg = [min(1.0, round(max(a, b) + 0.08 * (b > 0.2), 2)) for a, b in zip(agg, vec)]

    fig = go.Figure(
        go.Scatterpolar(
            r=agg + [agg[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Combined Dish Spectrum",
            line=dict(color="#10b981", width=2.5),
            fillcolor="rgba(16, 185, 129, 0.22)",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
            angularaxis=dict(tickfont=dict(size=11, color="#e2e8f0")),
        ),
        showlegend=False,
        margin=dict(l=45, r=45, t=25, b=25),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.cache_resource(show_spinner="Loading BGE Bi-Encoder & Cross-Encoder models...")
def load_engine() -> FlavorEngine:
    return FlavorEngine()


engine = load_engine()

with st.sidebar:
    st.title("🧪 FlavorVector")
    st.caption("Real-Time Culinary Substitution & Flavor-Pairing Engine")
    st.divider()
    st.metric("Compute Device", engine.device.upper())
    c1, c2 = st.columns(2)
    c1.metric("Substitutions", engine.sub_collection.count())
    c2.metric("Ingredients", engine.pair_collection.count())
    if st.button("🗑️ Clear Live Web Cache", use_container_width=True):
        for col in (engine.sub_collection, engine.pair_collection):
            live_ids = [i for i in col.get()["ids"] if i.startswith("live_")]
            if live_ids:
                col.delete(ids=live_ids)
        st.rerun()
    st.divider()
    st.markdown(
        "**Hybrid Retrieval Pipeline**\n"
        "1. Dense search via `BAAI/bge-base-en-v1.5`\n"
        "2. Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`)\n"
        "3. Real-time web synthesis + `ChromaDB` auto-indexing"
    )

tab_sub, tab_pair = st.tabs(["🔄 Functional Substitutions", "🧬 Flavor Pairing"])

with tab_sub:
    st.subheader("Find Role-Matched Ingredient Substitutes")
    query = st.text_input(
        "What are you trying to replace and why?",
        value="what can replace buttermilk and give the same tanginess in pancakes?",
    )
    col1, col2, col3, col4, col5 = st.columns([2, 1.2, 1.4, 0.9, 1.1])
    dish_context = col1.text_input("Dish Context (optional)", placeholder="e.g., pancakes, samosa, pickle")
    taste_filter = col2.selectbox("Taste Profile", TASTE_OPTIONS, key="sub_taste")
    cuisine_filter = col3.selectbox("Culinary Corner", CUISINE_OPTIONS, key="sub_cuisine")
    vegan_only = col4.toggle("Vegan Only", value=False)
    force_live_sub = col5.toggle("Force Live Fetch", value=False)

    if st.button("Search Substitutes", type="primary", use_container_width=True):
        with st.spinner("Searching local vectors and live culinary sources..."):
            hits, fetched_live = engine.search_substitutes(
                query=query,
                dish_context=dish_context.strip() or None,
                taste_filter=taste_filter,
                cuisine_filter=cuisine_filter,
                vegan_only=vegan_only,
                force_live=force_live_sub,
                top_k=4,
            )
        if fetched_live:
            st.success("🌐 Fetched live culinary data from the web and indexed it into ChromaDB!")
        else:
            st.info("⚡ Retrieved directly from local ChromaDB vector cache.")

        if not hits:
            st.warning("No matching substitutions found. Try toggling 'Force Live Fetch' or clearing filters.")
        for idx, hit in enumerate(hits):
            with st.container(border=True):
                left, mid, right = st.columns([3.2, 1.8, 0.9])
                badge = "🌱 Vegan" if hit["vegan"] else "🥛 Dairy/Animal"
                src_badge = "🌐 Live Web" if hit.get("source") == "live_web" else "🗄️ Local Atlas"
                left.markdown(f"### #{hit['rank']} — {hit['substitute']}")
                left.caption(
                    f"Replaces: **{hit['target']}** · {badge} · {src_badge} · "
                    f"Taste: `{hit['tastes']}` · Corner: **{hit['cuisine']}**"
                )
                left.code(hit["details"], language="yaml")
                mid.plotly_chart(
                    render_substitution_radar(hit),
                    use_container_width=True,
                    key=f"sub_radar_{idx}_{hit['rank']}",
                )
                right.metric("Match Score", f"{hit['relevance_score'] * 100:.1f}%")

with tab_pair:
    st.subheader("Discover Complementary Flavor Pairings")
    raw_ings = st.text_input(
        "Base Ingredients (comma-separated)",
        value="garlic, lemon, thyme",
    )
    p1, p2, p3, p4, p5 = st.columns([2, 1.2, 1.4, 0.9, 1.1])
    dish_style = p1.text_input("Dish Style (optional)", placeholder="e.g., roasted chicken, masala chai")
    target_taste = p2.selectbox("Target Taste", TASTE_OPTIONS, key="pair_taste")
    pair_cuisine = p3.selectbox("Culinary Corner", CUISINE_OPTIONS, key="pair_cuisine")
    top_k_pair = p4.slider("Results", min_value=1, max_value=12, value=6)
    force_live_pair = p5.toggle("Force Live Fetch", value=False, key="live_pair")

    if st.button("Find Pairings", type="primary", use_container_width=True):
        ing_list = [i.strip() for i in raw_ings.split(",") if i.strip()]
        if not ing_list:
            st.error("Enter at least one ingredient.")
        else:
            with st.spinner("Mining live co-occurrences and flavor vectors..."):
                suggestions, fetched_live = engine.suggest_pairings(
                    ingredients=ing_list,
                    dish_style=dish_style.strip() or None,
                    target_taste=target_taste,
                    cuisine_filter=pair_cuisine,
                    force_live=force_live_pair,
                    top_k=top_k_pair,
                )
            if fetched_live:
                st.success("🌐 Fetched live ingredient co-occurrences and updated ChromaDB!")

            if suggestions:
                st.markdown("#### 🕸️ Combined 10-Taste Dish Spectrum")
                st.plotly_chart(render_pairing_radar(suggestions), use_container_width=True, key="pair_spectrum")

            for item in suggestions:
                with st.container(border=True):
                    left, right = st.columns([4, 1])
                    left.markdown(f"### #{item['rank']} — {item['ingredient'].title()}")
                    left.caption(f"Taste: `{item['tastes']}` · Corner: **{item['cuisine']}**")
                    if item["common_partners"]:
                        left.markdown(f"**Co-occurs with:** {item['common_partners']}")
                    if item["compounds"]:
                        left.markdown(f"**Key Compounds:** `{item['compounds']}`")
                    right.metric("Affinity", f"{item['relevance_score'] * 100:.1f}%")
                    with st.expander("Full Semantic Profile"):
                        st.text(item["details"])
