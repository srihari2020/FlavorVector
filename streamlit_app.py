import streamlit as st
from app.engine import FlavorEngine

st.set_page_config(
    page_title="FlavorVector | Food-Science Search",
    page_icon="🧪",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading BGE Bi-Encoder & Cross-Encoder models...")
def load_engine() -> FlavorEngine:
    return FlavorEngine()


engine = load_engine()

with st.sidebar:
    st.title("🧪 FlavorVector")
    st.caption("Semantic Cooking Substitution & Flavor-Pairing Engine")
    st.divider()
    st.metric("Compute Device", engine.device.upper())
    col1, col2 = st.columns(2)
    col1.metric("Substitutions", engine.sub_collection.count())
    col2.metric("Ingredients", engine.pair_collection.count())
    st.divider()
    st.markdown(
        "**Pipeline**\n"
        "- Dense Retrieval: `BAAI/bge-base-en-v1.5`\n"
        "- Reranker: `ms-marco-MiniLM-L-6-v2`\n"
        "- Vector Store: `ChromaDB`"
    )

tab_sub, tab_pair = st.tabs(["🔄 Functional Substitutions", "🧬 Flavor Pairing"])

with tab_sub:
    st.subheader("Find Role-Matched Ingredient Substitutes")
    query = st.text_input(
        "What are you trying to replace and why?",
        value="what can replace buttermilk and give the same tanginess in pancakes?",
    )
    c1, c2, c3 = st.columns([2, 1, 1])
    dish_context = c1.text_input("Dish Context (optional)", placeholder="e.g., fluffy pancakes, meatballs")
    vegan_only = c2.toggle("Vegan Only", value=False)
    top_k_sub = c3.slider("Results", min_value=1, max_value=6, value=3)

    if st.button("Search Substitutes", type="primary", use_container_width=True):
        hits = engine.search_substitutes(
            query=query,
            dish_context=dish_context.strip() or None,
            vegan_only=vegan_only,
            top_k=top_k_sub,
        )
        if not hits:
            st.warning("No matching substitutions found. Try disabling filters or indexing more rules.")
        for hit in hits:
            with st.container(border=True):
                left, right = st.columns([4, 1])
                badge = "🌱 Vegan" if hit["vegan"] else "🥛 Dairy/Animal"
                left.markdown(f"### #{hit['rank']} — {hit['substitute']}")
                left.caption(f"Replaces: **{hit['target']}** · {badge}")
                right.metric("Match Score", f"{hit['relevance_score'] * 100:.1f}%")
                st.code(hit["details"], language="yaml")

with tab_pair:
    st.subheader("Discover Complementary Flavor Pairings")
    raw_ings = st.text_input(
        "Base Ingredients (comma-separated)",
        value="garlic, lemon, thyme",
    )
    p1, p2 = st.columns([3, 1])
    dish_style = p1.text_input("Dish Style (optional)", placeholder="e.g., roasted vegetables, spicy broth")
    top_k_pair = p2.slider("Suggestions", min_value=1, max_value=12, value=5)

    if st.button("Find Pairings", type="primary", use_container_width=True):
        ing_list = [i.strip() for i in raw_ings.split(",") if i.strip()]
        if not ing_list:
            st.error("Enter at least one ingredient.")
        else:
            suggestions = engine.suggest_pairings(
                ingredients=ing_list,
                dish_style=dish_style.strip() or None,
                top_k=top_k_pair,
            )
            if not suggestions:
                st.warning("No pairings indexed yet. Run `python3 scripts/build_pairings.py` first.")
            for item in suggestions:
                with st.container(border=True):
                    left, right = st.columns([4, 1])
                    left.markdown(f"### #{item['rank']} — {item['ingredient'].title()}")
                    if item["common_partners"]:
                        left.markdown(f"**Co-occurs with:** {item['common_partners']}")
                    if item["compounds"]:
                        left.markdown(f"**Key Compounds:** `{item['compounds']}`")
                    right.metric("Affinity", f"{item['relevance_score'] * 100:.1f}%")
                    with st.expander("Full Semantic Profile"):
                        st.text(item["details"])
