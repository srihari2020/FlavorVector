import streamlit as st
from app.engine import FlavorEngine

st.set_page_config(
    page_title="FlavorVector | Real-Time Food-Science Search",
    page_icon="🧪",
    layout="wide",
)

TASTE_OPTIONS = [
    "All Tastes",
    "sweet",
    "salty",
    "sour",
    "bitter",
    "umami",
    "spicy_pungent",
    "astringent",
    "cooling",
    "fatty_rich",
    "kokumi",
]

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
        "**Hybrid Retrieval Mode**\n"
        "1. Checks local `ChromaDB` vectors\n"
        "2. Fetches live from Wikipedia, TheMealDB & Open Food Facts on cache miss\n"
        "3. Auto-embeds & persists new cards"
    )

tab_sub, tab_pair = st.tabs(["🔄 Functional Substitutions", "🧬 Flavor Pairing"])

with tab_sub:
    st.subheader("Find Role-Matched Ingredient Substitutes")
    query = st.text_input(
        "What are you trying to replace and why?",
        value="what can I use instead of green chutney in a samosa",
    )
    col1, col2, col3, col4, col5 = st.columns([2, 1.2, 1.4, 0.9, 1.1])
    dish_context = col1.text_input("Dish Context (optional)", placeholder="e.g., samosa, banana shake, dosa")
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
        for hit in hits:
            with st.container(border=True):
                left, right = st.columns([4, 1])
                badge = "🌱 Vegan" if hit["vegan"] else "🥛 Dairy/Animal"
                src_badge = "🌐 Live Web" if hit.get("source") == "live_web" else "🗄️ Local Atlas"
                left.markdown(f"### #{hit['rank']} — {hit['substitute']}")
                left.caption(
                    f"Replaces: **{hit['target']}** · {badge} · {src_badge} · "
                    f"Taste: `{hit['tastes']}` · Corner: **{hit['cuisine']}**"
                )
                right.metric("Match Score", f"{hit['relevance_score'] * 100:.1f}%")
                st.code(hit["details"], language="yaml")

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
