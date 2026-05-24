"""
frontend/app.py
================
NaijaNutri Pro — Streamlit chat interface.

Onboarding: 4 steps
  Step 1 — Basic profile (archetype, language)
  Step 2 — Food profile (dietary, weight goal, TDEE, food budget, location)
  Step 3 — Products profile (interests, product budget)
  Step 4 — Books profile (genres, book budget)

Main app: 4 tabs — each uses ONLY its own domain context.
  Food:     dietary + TDEE + weight goal + location + food budget
  Products: product interests + product budget only
  Books:    book genres + book budget only
"""

import os
import uuid
import httpx
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="NaijaNutri Pro",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Make the chat bubbles look more like iMessage/WhatsApp */
        .stChatMessage {
            border-radius: 15px;
            padding: 10px;
        }
        
        /* Add a subtle glow to primary buttons */
        button[kind="primary"] {
            box-shadow: 0 4px 14px 0 rgba(0, 135, 81, 0.39);
        }
    </style>
""", unsafe_allow_html=True)

# ─── Session state defaults ───────────────────────────────────────────────────

defaults = {
    "user_id":              f"user_{uuid.uuid4().hex[:8]}",
    "onboarded":            False,
    "onboard_step":         1,
    # Basic
    "archetype":            "working_class",
    "language":             "english",
    # Food profile
    "food_budget":          3000,
    "dietary_restrictions": [],
    "weight_goal":          "maintain",
    "tdee":                 None,
    "calories_per_meal":    None,
    "meals_per_day":        3,
    "location":             "",
    "liked_food_cats":      [],
    # Products profile
    "product_budget":       50000,
    "product_interests":    [],
    # Books profile
    "book_budget":          5000,
    "book_genres":          [],
    # Chat histories — completely separate
    "chat_food":            [],
    "chat_products":        [],
    "chat_books":           [],
    # Feedback
    "liked_items":          [],
    "disliked_items":       [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── TDEE helpers ─────────────────────────────────────────────────────────────

def calculate_tdee(weight_kg, height_cm, age, sex, activity, meals):
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age
    bmr += 5 if sex == "Male" else -161
    factors = {
        "Sedentary (desk job, little/no exercise)":     1.2,
        "Lightly active (1–3 days/week)":               1.375,
        "Moderately active (3–5 days/week)":            1.55,
        "Very active (6–7 days/week)":                  1.725,
        "Extra active (physical job + daily exercise)": 1.9,
    }
    tdee = int(bmr * factors.get(activity, 1.375))
    return int(bmr), tdee, tdee // meals


def adjusted_target(tdee: int, goal: str) -> int:
    if goal == "lose":   return max(1200, tdee - 500)
    elif goal == "gain": return tdee + 300
    return tdee


# ══════════════════════════════════════════════════════════════════════════════
# ONBOARDING
# ══════════════════════════════════════════════════════════════════════════════

def render_onboarding():
    col = st.columns([1, 2, 1])[1]
    with col:
        st.image("https://flagcdn.com/w80/ng.png", width=48)
        st.title("NaijaNutri Pro 🇳🇬")
        st.caption("Quick setup — takes about 2 minutes.")
        st.divider()

        step = st.session_state.onboard_step
        labels = ["Basic Profile", "Food & Health", "Products", "Books"]
        prog = st.columns(4)
        for i, label in enumerate(labels):
            with prog[i]:
                if i + 1 < step:
                    st.markdown(f"✅ ~~{label}~~")
                elif i + 1 == step:
                    st.markdown(f"**→ {label}**")
                else:
                    st.markdown(f"<span style='color:gray'>{label}</span>",
                                unsafe_allow_html=True)
        st.divider()

        # ── Step 1: Basic ─────────────────────────────────────────────────────
        if step == 1:
            st.subheader("👤 Step 1: Basic Profile")

            archetype = st.selectbox(
                "Who are you?",
                options=["sapa_student","hustling_corper","working_class",
                         "tech_bro","omo_landlord","naija_mama"],
                format_func=lambda x: {
                    "sapa_student":"🎓 Student — Smart & Frugal",
                    "hustling_corper": "🟢 Hustling Corper — NYSC life",
                    "working_class":   "💼 Working Class — 9-to-5 grind",
                    "tech_bro":        "💻 Tech Bro/Sis — aesthetic & vibes",
                    "omo_landlord":    "💰 Omo Landlord — premium only",
                    "naija_mama":      "🍲 Naija Mama — home standards",
                }[x],
            )

            language = st.radio(
                "How should I talk to you?",
                ["english", "pidgin"],
                format_func=lambda x: (
                    "🇬🇧 Standard English" if x == "english"
                    else "🇳🇬 Nigerian Pidgin — e go dey smooth"
                ),
                horizontal=True,
            )
            st.divider()
            if st.button("Next →", use_container_width=True, type="primary"):
                st.session_state.archetype = archetype
                st.session_state.language  = language
                st.session_state.onboard_step = 2
                st.rerun()

        # ── Step 2: Food & Health ─────────────────────────────────────────────
        elif step == 2:
            st.subheader("🍽️ Step 2: Food & Health")
            st.caption("This section ONLY affects food recommendations.")

            food_budget = st.number_input(
                "Food budget per meal (₦)",
                min_value=100, max_value=50_000, value=3_000, step=500,
            )

            location = st.text_input(
                "Your city / area (optional)",
                placeholder="e.g. Lagos Island, Abuja, Ibadan",
                help="Helps us prioritise nearby restaurants",
            )

            dietary_restrictions = st.multiselect(
                "Dietary restrictions or goals",
                ["vegetarian","vegan","halal","no pork","no seafood",
                 "gluten free","no dairy","high protein","low carb","low calorie"],
                help="Hard restrictions (vegetarian, halal…) filter out non-compliant items. "
                     "Goals (high protein…) influence ranking.",
            )

            weight_goal = st.radio(
                "Weight goal",
                ["lose","maintain","gain"],
                format_func=lambda x: {
                    "lose":     "📉 Lose weight  (−500 kcal/day deficit)",
                    "maintain": "⚖️  Maintain weight",
                    "gain":     "📈 Gain weight  (+300 kcal/day surplus)",
                }[x],
                index=1,
            )

            liked_food = st.multiselect(
                "Favourite food categories",
                ["Nigerian Food","Fast Food","Healthy Food","Grills",
                 "Snacks","Seafood","Vegetarian","Cafés"],
            )

            st.divider()
            st.subheader("🔥 TDEE Calculator (optional)")
            st.caption("Mifflin-St Jeor — we'll use this to recommend right-sized meals.")

            with st.expander("Calculate my daily calorie needs"):
                wc1, wc2 = st.columns(2)
                with wc1:
                    weight  = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, 0.5, key="ob_w")
                    age_val = st.number_input("Age", 15, 90, 25, 1, key="ob_a")
                with wc2:
                    height = st.number_input("Height (cm)", 100.0, 250.0, 170.0, 0.5, key="ob_h")
                    sex    = st.selectbox("Sex", ["Male","Female"], key="ob_s")

                activity = st.selectbox(
                    "Activity level",
                    ["Sedentary (desk job, little/no exercise)",
                     "Lightly active (1–3 days/week)",
                     "Moderately active (3–5 days/week)",
                     "Very active (6–7 days/week)",
                     "Extra active (physical job + daily exercise)"],
                    index=1, key="ob_act",
                )
                meals_val = st.slider("Meals per day", 2, 6, 3, key="ob_m")

                if st.button("⚡ Calculate TDEE", use_container_width=True):
                    bmr, tdee, _ = calculate_tdee(
                        weight, height, int(age_val), sex, activity, meals_val
                    )
                    adj = adjusted_target(tdee, weight_goal)
                    st.session_state.tdee           = tdee
                    st.session_state.meals_per_day  = meals_val
                    st.session_state.calories_per_meal = adj // meals_val
                    icon = {"lose":"📉","gain":"📈","maintain":"⚖️"}[weight_goal]
                    st.success(f"**BMR:** {bmr} kcal/day")
                    st.info(f"**TDEE:** {tdee} kcal/day")
                    st.info(f"{icon} **Adjusted target:** {adj} kcal/day → ~{adj//meals_val} kcal/meal")

            if st.session_state.tdee:
                adj = adjusted_target(st.session_state.tdee, weight_goal)
                st.success(f"🔥 TDEE set: **{st.session_state.tdee}** kcal/day → target **{adj}** kcal/day")

            st.divider()
            col_b, col_n = st.columns(2)
            with col_b:
                if st.button("← Back", use_container_width=True):
                    st.session_state.onboard_step = 1
                    st.rerun()
            with col_n:
                if st.button("Next →", use_container_width=True, type="primary"):
                    st.session_state.food_budget          = food_budget
                    st.session_state.location             = location
                    st.session_state.dietary_restrictions = dietary_restrictions
                    st.session_state.weight_goal          = weight_goal
                    st.session_state.liked_food_cats      = liked_food
                    st.session_state.onboard_step = 3
                    st.rerun()

        # ── Step 3: Products ──────────────────────────────────────────────────
        elif step == 3:
            st.subheader("📦 Step 3: Products")
            st.caption("This section ONLY affects product recommendations — no food context here.")

            product_budget = st.number_input(
                "Product budget (₦)",
                min_value=1_000, max_value=2_000_000, value=50_000, step=5_000,
                help="e.g. ₦50,000 for accessories, ₦200,000+ for laptops/phones",
            )

            product_interests = st.multiselect(
                "What kind of appliances do you usually shop for?",
                ["Microwaves & Ovens", "Refrigerators & Freezers", 
                 "Washing Machines & Dryers", "Air Conditioners & Fans", 
                 "Vacuums & Floor Care", "Small Kitchen Appliances", 
                 "Water Heaters & Dispensers", "Generators & Power Solutions"],
            )

            st.divider()
            col_b, col_n = st.columns(2)
            with col_b:
                if st.button("← Back", use_container_width=True, key="p_back"):
                    st.session_state.onboard_step = 2
                    st.rerun()
            with col_n:
                if st.button("Next →", use_container_width=True, type="primary", key="p_next"):
                    st.session_state.product_budget    = product_budget
                    st.session_state.product_interests = product_interests
                    st.session_state.onboard_step = 4
                    st.rerun()

        # ── Step 4: Books ─────────────────────────────────────────────────────
        elif step == 4:
            st.subheader("📚 Step 4: Books")
            st.caption("This section ONLY affects book recommendations — no food context here.")

            book_budget = st.number_input(
                "Book budget (₦)",
                min_value=500, max_value=100_000, value=5_000, step=500,
            )

            book_genres = st.multiselect(
                "Favourite genres",
                ["Nigerian Fiction","African Literature","Self-Help","Business & Finance",
                 "Biography & Memoir","Science Fiction","Romance","Thriller",
                 "History","Productivity","Spirituality","Health & Wellness"],
            )

            st.divider()
            col_b, col_n = st.columns(2)
            with col_b:
                if st.button("← Back", use_container_width=True, key="bk_back"):
                    st.session_state.onboard_step = 3
                    st.rerun()
            with col_n:
                if st.button("Finish Setup 🚀", use_container_width=True,
                             type="primary", key="bk_next"):
                    st.session_state.book_budget = book_budget
                    st.session_state.book_genres = book_genres

                    # Save persona to backend
                    try:
                        httpx.post(
                            f"{BACKEND}/persona",
                            json={
                                "user_id":          st.session_state.user_id,
                                "archetype_key":    st.session_state.archetype,
                                "budget_naira":     int(st.session_state.food_budget),
                                "dietary_notes":    ", ".join(st.session_state.dietary_restrictions),
                                "liked_categories": st.session_state.liked_food_cats,
                            },
                            timeout=15,
                        )
                    except Exception:
                        pass

                    st.session_state.onboarded = True
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def render_main_app():

    with st.sidebar:
        st.image("https://flagcdn.com/w80/ng.png", width=32)
        st.title("NaijaNutri Pro")
        st.caption(f"Session: `{st.session_state.user_id}`")
        st.divider()

        arch_label = {
            "sapa_student":"🎓 Sapa Student","hustling_corper":"🟢 Corper",
            "working_class":"💼 Working Class","tech_bro":"💻 Tech Bro/Sis",
            "omo_landlord":"💰 Omo Landlord","naija_mama":"🍲 Naija Mama",
        }.get(st.session_state.archetype, "")

        st.markdown(f"**{arch_label}**")
        st.caption(f"Language: {st.session_state.language.title()}")

        st.divider()
        st.caption("**Food**")
        st.caption(f"Budget: ₦{st.session_state.food_budget:,}")
        if st.session_state.location:
            st.caption(f"📍 {st.session_state.location}")
        if st.session_state.dietary_restrictions:
            st.caption("Diet: " + ", ".join(st.session_state.dietary_restrictions))
        if st.session_state.tdee:
            adj = adjusted_target(st.session_state.tdee, st.session_state.weight_goal)
            cpm = adj // st.session_state.meals_per_day
            icon = {"lose":"📉","gain":"📈","maintain":"⚖️"}[st.session_state.weight_goal]
            st.caption(f"{icon} {adj} kcal/day · ~{cpm}/meal")

        st.divider()
        st.caption("**Products**")
        st.caption(f"Budget: ₦{st.session_state.product_budget:,}")
        if st.session_state.product_interests:
            st.caption(", ".join(st.session_state.product_interests[:3]))

        st.divider()
        st.caption("**Books**")
        st.caption(f"Budget: ₦{st.session_state.book_budget:,}")
        if st.session_state.book_genres:
            st.caption(", ".join(st.session_state.book_genres[:3]))

        st.divider()
        if st.session_state.liked_items or st.session_state.disliked_items:
            st.divider()
            st.caption("**Session feedback**")
            if st.session_state.liked_items:
                st.success(f"👍 **{len(st.session_state.liked_items)}** liked item(s)")
            if st.session_state.disliked_items:
                st.error(f"👎 **{len(st.session_state.disliked_items)}** item(s) excluded from future results")
            if st.button("Clear feedback", use_container_width=True):
                st.session_state.liked_items    = []
                st.session_state.disliked_items = []
                st.rerun()

        if st.button("🗑 Clear all chats", use_container_width=True):
            st.session_state.chat_food     = []
            st.session_state.chat_products = []
            st.session_state.chat_books    = []
            st.rerun()

        if st.button("↩️ Redo onboarding", use_container_width=True):
            st.session_state.onboarded    = False
            st.session_state.onboard_step = 1
            st.rerun()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    st.title("NaijaNutri Pro")

    tab_food, tab_products, tab_books, tab_sim = st.tabs([
        "🍽️ Nigerian Dishes",
        "📦 Products",
        "📚 Books",
        "📝 Review Simulator",
    ])

    # ── Nigerian Dishes tab ───────────────────────────────────────────────────
    with tab_food:
        col_c, col_o = st.columns([3, 1])
        with col_o:
            st.subheader("⚙️ Options")
            food_n    = st.slider("Recommendations", 3, 15, 5, key="food_n")
            plan_mode = st.toggle("🗓 5-day meal plan")

            st.divider()
            st.caption("**Meal type**")
            meal_type_filter = st.radio(
                "I want to…",
                ["Any", "🛒 Buy from vendor", "🍳 Quick home cook", "👨‍🍳 Proper cooking"],
                index=0,
                key="meal_type_filter",
                help="Vendor: ready to eat in <15 mins. Quick: 15-30 mins. Proper: 30+ mins.",
            )

            meal_type_map = {
                "Any":                  None,
                "🛒 Buy from vendor":   "buy_from_vendor",
                "🍳 Quick home cook":   "quick_home_cook",
                "👨‍🍳 Proper cooking":   "home_cooking",
            }
            selected_meal_type = meal_type_map[meal_type_filter]

            st.divider()
            if st.session_state.tdee:
                adj = adjusted_target(st.session_state.tdee, st.session_state.weight_goal)
                cpm = adj // st.session_state.meals_per_day
                icon = {"lose":"📉","gain":"📈","maintain":"⚖️"}[st.session_state.weight_goal]
                st.success(f"{icon} **{adj} kcal/day**\n\n~{cpm} kcal/meal")
            else:
                st.info("💡 Set TDEE in onboarding for calorie-aware suggestions")

            if st.session_state.dietary_restrictions:
                st.warning(
                    "**Active filters:**\n\n" +
                    "\n".join(f"• {r}" for r in st.session_state.dietary_restrictions)
                )

        with col_c:
            st.caption(
                "Recommending Nigerian dishes — from quick street food to full home-cooked meals. "
                "Costs are estimated vendor prices or ingredient costs."
            )
            render_chat("chat_food", show_meal_plan=True)

            # Build meal type hint for query
            meal_hint = ""
            if selected_meal_type == "buy_from_vendor":
                meal_hint = " quick street food vendor"
            elif selected_meal_type == "quick_home_cook":
                meal_hint = " quick easy to cook"
            elif selected_meal_type == "home_cooking":
                meal_hint = " proper home cooked meal"

            food_q = st.chat_input(
                "e.g. 'high protein Yoruba swallow under ₦2000' or 'light vegan snack'",
                key="in_food",
            )
            if food_q:
                adj_cal = (
                    adjusted_target(st.session_state.tdee, st.session_state.weight_goal)
                    if st.session_state.tdee else None
                )
                handle_query(
                    query=food_q + meal_hint,
                    chat_key="chat_food",
                    domains=["nigerian_food"],
                    n=food_n,
                    plan_mode=plan_mode,
                    show_meal_plan=True,
                    budget=st.session_state.food_budget,
                    dietary_restrictions=st.session_state.dietary_restrictions,
                    calorie_target=adj_cal,
                    weight_goal=st.session_state.weight_goal,
                    location=st.session_state.location,
                    product_interests=[],
                    book_genres=[],
                )

    # ── Products tab ──────────────────────────────────────────────────────────
    with tab_products:
        col_c2, col_o2 = st.columns([3, 1])
        with col_o2:
            st.subheader("⚙️ Options")
            prod_n = st.slider("Recommendations", 3, 15, 5, key="prod_n")
            st.metric("Budget", f"₦{st.session_state.product_budget:,}")
            if st.session_state.product_interests:
                st.caption("Interests: " + ", ".join(st.session_state.product_interests))

        with col_c2:
            render_chat("chat_products", show_meal_plan=False)
            prod_q = st.chat_input(
                "e.g. 'best budget microwave' or 'energy-saving fridge under ₦150k'",
                key="in_prod"
            )
            if prod_q:
                handle_query(
                    query=prod_q, chat_key="chat_products",
                    domains=["amazon"], n=prod_n,
                    plan_mode=False, show_meal_plan=False,
                    budget=st.session_state.product_budget,
                    dietary_restrictions=[], calorie_target=None,
                    weight_goal="maintain", location="",
                    product_interests=st.session_state.product_interests,
                    book_genres=[],
                )

    # ── Books tab ─────────────────────────────────────────────────────────────
    with tab_books:
        col_c3, col_o3 = st.columns([3, 1])
        with col_o3:
            st.subheader("⚙️ Options")
            book_n = st.slider("Recommendations", 3, 10, 5, key="book_n")
            st.metric("Budget", f"₦{st.session_state.book_budget:,}")
            if st.session_state.book_genres:
                st.caption("Genres: " + ", ".join(st.session_state.book_genres))

        with col_c3:
            render_chat("chat_books", show_meal_plan=False)
            book_q = st.chat_input(
                "e.g. 'motivational book for an entrepreneur' or 'Nigerian fiction'",
                key="in_books"
            )
            if book_q:
                handle_query(
                    query=book_q, chat_key="chat_books",
                    domains=["goodreads"], n=book_n,
                    plan_mode=False, show_meal_plan=False,
                    budget=st.session_state.book_budget,
                    dietary_restrictions=[], calorie_target=None,
                    weight_goal="maintain", location="",
                    product_interests=[],
                    book_genres=st.session_state.book_genres,
                )

    # ── Review Simulator ──────────────────────────────────────────────────────
    with tab_sim:
        st.subheader("📝 Review Simulator")
        st.caption(
            "Task A: Predict how a specific Nigerian user would rate and review an item. "
            "Use **Single** mode to simulate one user, or **Compare Archetypes** to see "
            "all 6 Nigerian personas react to the same item."
        )

        sim_mode = st.radio(
            "Mode",
            ["Single user simulation", "Compare all 6 archetypes"],
            horizontal=True,
        )
        st.divider()

        # ── Shared item inputs ────────────────────────────────────────────────
        ic1, ic2, ic3 = st.columns(3)
        
        with ic1:
            # 1. User-Friendly Domain Dropdown
            domain_display_map = {
                "🍽️ Food & Restaurants": "yelp",
                "📦 Products & Appliances": "amazon",
                "📚 Books & Literature": "goodreads"
            }
            selected_domain_label = st.selectbox("Domain", list(domain_display_map.keys()))
            
            # This keeps compatibility with your backend variable!
            sim_domain = domain_display_map[selected_domain_label] 
            
            # 2. Dynamic Placeholders based on domain
            if sim_domain == "yelp":
                item_ph = "e.g. Chicken Republic Ibadan"
                cat_ph = "e.g. Fast Food"
                price_label = "Est. Price (₦) optional"
            elif sim_domain == "amazon":
                item_ph = "e.g. LG 20L Microwave Oven"
                cat_ph = "e.g. Home Appliances"
                price_label = "Est. Price (₦) optional"
            else: # goodreads
                item_ph = "e.g. Children of Blood and Bone"
                cat_ph = "e.g. Young Adult Fantasy"
                price_label = "Book Price (₦) optional"
                
            sim_name = st.text_input("Item name *", placeholder=item_ph)
            
        with ic2:
            sim_category  = st.text_input("Category", placeholder=cat_ph)
            sim_price     = st.number_input(price_label, min_value=0, value=0, step=500)
            sim_id        = st.text_input("Item ID (optional)", placeholder="auto-generated if blank")
            
        with ic3:
            apply_adapter = st.toggle("Nigerian language adapter", value=True)
            st.caption(
                "ON = Pidgin/slang adapted output (demo mode). "
                "OFF = plain English (used for ROUGE/BERTScore evaluation)."
            )
        # ── Single mode ───────────────────────────────────────────────────────
        if sim_mode == "Single user simulation":
            sim_user = st.text_input("User ID", value=st.session_state.user_id)

            if st.button("🎭 Simulate Review", use_container_width=True, type="primary"):
                if not sim_name:
                    st.warning("Please enter an item name.")
                else:
                    with st.spinner("Building persona and simulating review…"):
                        # Build metadata payload
                        meta_payload = {}
                        if sim_price > 0:
                            meta_payload["estimated_price_naira"] = sim_price

                        try:
                            resp = httpx.post(
                                f"{BACKEND}/simulate",
                                json={
                                    "user_id":                sim_user, # <--- MUST BE sim_user
                                    "item_id":                sim_id or f"item_{sim_name.replace(' ','_').lower()}",
                                    "item_name":              sim_name,
                                    "item_domain":            sim_domain,
                                    "item_category":          sim_category or "General",
                                    "item_metadata":          meta_payload,
                                    "apply_nigerian_adapter": apply_adapter,
                                },
                                timeout=120,
                            )
                            resp.raise_for_status()
                            d = resp.json()

                            st.divider()

                            # Metrics row
                            m1, m2, m3 = st.columns(3)
                            rating = d["predicted_rating"]
                            stars  = "⭐" * round(rating)
                            with m1:
                                st.metric("Predicted Rating", f"{rating}/5.0")
                                st.caption(stars)
                            with m2:
                                st.metric("Archetype Detected", d["archetype_name"])
                            with m3:
                                sentiment = (
                                    "Positive 😊" if rating >= 4.0 else
                                    "Mixed 😐"    if rating >= 2.5 else
                                    "Negative 😞"
                                )
                                st.metric("Sentiment", sentiment)

                            # Review
                            st.subheader("Generated Review")
                            st.info(f"*\"{ d['review_text'] }\"*")

                            col_r, col_raw = st.columns(2)
                            with col_r:
                                with st.expander("💡 Rating reasoning"):
                                    st.write(d["reasoning"])
                            with col_raw:
                                if apply_adapter:
                                    with st.expander("📄 Raw review (for ROUGE eval)"):
                                        st.write(d["raw_review_text"])

                            # Add to history
                            if "sim_history" not in st.session_state:
                                st.session_state.sim_history = []
                            st.session_state.sim_history.insert(0, {
                                "item":     sim_name,
                                "domain":   sim_domain,
                                "rating":   rating,
                                "archetype":d["archetype_name"],
                                "review":   d["review_text"][:120] + "…",
                            })
                            st.session_state.sim_history = st.session_state.sim_history[:10]

                        except Exception as e:
                            st.error(f"Error: {e}")

        # ── Compare archetypes mode ───────────────────────────────────────────
        else:
            st.caption(
                "Simulates the same item across all 6 Nigerian archetypes. "
                "Great for demonstrating behavioural diversity in your solution paper."
            )

            ARCHETYPE_USERS = {
                "sapa_student":    ("🎓 Student — Smart & Frugal", "demo_user_sapa"),
                "hustling_corper": ("🟢 Hustling Corper", "demo_user_corper"),
                "working_class":   ("💼 Working Class",   "demo_user_wc"),
                "tech_bro":        ("💻 Tech Bro/Sis",    "demo_user_tech"),
                "omo_landlord":    ("💰 Omo Landlord",    "demo_user_omo"),
                "naija_mama":      ("🍲 Naija Mama",      "demo_user_mama"),
            }

            if st.button("🎭 Compare All 6 Archetypes", use_container_width=True, type="primary"):
                if not sim_name:
                    st.warning("Please enter an item name.")
                else:
                    results = {}
                    progress = st.progress(0, text="Simulating…")
                    for i, (arch_key, (arch_label, user_id)) in enumerate(ARCHETYPE_USERS.items()):
                        progress.progress(
                            (i + 1) / len(ARCHETYPE_USERS),
                            text=f"Simulating {arch_label}…"
                        )
                        # Build metadata payload
                        meta_payload = {}
                        if sim_price > 0:
                            meta_payload["estimated_price_naira"] = sim_price

                        try:
                            resp = httpx.post(
                                f"{BACKEND}/simulate",
                                json={
                                    "user_id":                user_id,
                                    "item_id":                sim_id or f"item_{sim_name.replace(' ','_').lower()}",
                                    "item_name":              sim_name,
                                    "item_domain":            sim_domain,
                                    "item_category":          sim_category or "General",
                                    "item_metadata":          meta_payload,
                                    "apply_nigerian_adapter": apply_adapter,
                                },
                                timeout=120,
                            )
                            resp.raise_for_status()
                            results[arch_key] = resp.json()
                        except Exception as e:
                            results[arch_key] = {"error": str(e)}

                    progress.empty()
                    st.divider()
                    st.subheader(f"How 6 Nigerians would review: **{sim_name}**")

                    # Rating summary bar
                    valid = {k: v for k, v in results.items() if "error" not in v}
                    if valid:
                        avg_r = sum(v["predicted_rating"] for v in valid.values()) / len(valid)
                        st.caption(
                            f"Average predicted rating: **{avg_r:.1f}/5.0** across {len(valid)} archetypes"
                        )

                    # Cards per archetype
                    for arch_key, (arch_label, _) in ARCHETYPE_USERS.items():
                        d = results.get(arch_key, {})
                        if "error" in d:
                            st.error(f"{arch_label}: {d['error']}")
                            continue

                        rating = d["predicted_rating"]
                        stars  = "⭐" * round(rating)
                        sentiment_color = (
                            "✅" if rating >= 4.0 else
                            "🟡" if rating >= 2.5 else
                            "🔴"
                        )

                        with st.container(border=True):
                            h1, h2 = st.columns([3, 1])
                            with h1:
                                st.markdown(f"**{arch_label}**")
                                st.caption(f"{stars} {rating}/5.0")
                            with h2:
                                st.markdown(f"### {sentiment_color}")
                            st.markdown(f"*\"{d['review_text']}\"*")
                            with st.expander("Reasoning"):
                                st.caption(d.get("reasoning", ""))

        # ── Simulation history ────────────────────────────────────────────────
        if "sim_history" in st.session_state and st.session_state.sim_history:
            st.divider()
            st.subheader("📋 Recent simulations")
            for h in st.session_state.sim_history:
                st.caption(
                    f"**{h['item']}** ({h['domain']}) · "
                    f"{h['archetype']} · ⭐ {h['rating']}/5.0 — {h['review']}"
                )

    st.divider()
    st.caption(
        "NaijaNutri Pro · DSN × BCT Hackathon 3.0 · "
        "Llama 3.1 (Groq) · ChromaDB · Sentence Transformers · Mifflin-St Jeor TDEE"
    )


# ─── Shared helpers ───────────────────────────────────────────────────────────

def render_chat(chat_key: str, show_meal_plan: bool):
    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

            if msg.get("calories_per_meal"):
                st.caption(f"🔥 Calibrated for ~{msg['calories_per_meal']} kcal/meal")

            for i, rec in enumerate(msg.get("recommendations", [])):
                with st.container(border=True):
                    ci, cf = st.columns([5, 1])
                    with ci:
                        emoji = {"yelp":"🍽️","amazon":"📦","goodreads":"📚"}.get(
                            rec["item_domain"], "🔷"
                        )
                        stars = "⭐" * round(rec["avg_rating"])
                        st.markdown(
                            f"**{i+1}. {emoji} {rec['item_name']}** {stars} {rec['avg_rating']:.1f}"
                        )
                        st.caption(f"🏷️ **Category:** {rec.get('item_category', 'General')}")
                        if rec.get("calorie_note"):
                            st.caption(f"🔥 {rec['calorie_note']}")
                    with cf:
                        fb = f"{msg.get('msg_id','x')}_{i}"
                        if st.button("👍", key=f"like_{fb}"):
                            if rec["item_id"] not in st.session_state.liked_items:
                                st.session_state.liked_items.append(rec["item_id"])
                        if st.button("👎", key=f"dislike_{fb}"):
                            if rec["item_id"] not in st.session_state.disliked_items:
                                st.session_state.disliked_items.append(rec["item_id"])

            if show_meal_plan and msg.get("meal_plan"):
                st.subheader("🗓 5-Day Meal Plan")
                for day in msg["meal_plan"]:
                    note = f"  \n🔥 {day['calorie_note']}" if day.get("calorie_note") else ""
                    st.markdown(
                        f"**{day['day']}**: {day['item_name']}  \n_{day['reason']}_{note}"
                    )


def handle_query(
    query, chat_key, domains, n, plan_mode, show_meal_plan,
    budget, dietary_restrictions, calorie_target, weight_goal,
    location, product_interests, book_genres,
):
    msg_id = uuid.uuid4().hex[:6]
    st.session_state[chat_key].append(
        {"role": "user", "content": query, "msg_id": msg_id}
    )
    with st.spinner("Finding options for you…"):
        try:
            resp = httpx.post(
                f"{BACKEND}/recommend",
                json={
                    "user_id":              st.session_state.user_id,
                    "query":                query,
                    "n":                    n,
                    "domains":              domains,
                    "budget_naira":         int(budget),
                    "dietary_restrictions": dietary_restrictions,
                    "calorie_target":       calorie_target,
                    "meals_per_day":        st.session_state.meals_per_day,
                    "weight_goal":          weight_goal,
                    "location":             location,
                    "product_interests":    product_interests,
                    "book_genres":          book_genres,
                    "language_preference":  st.session_state.language,
                    "plan_mode":            plan_mode,
                    "exclude_items":        st.session_state.disliked_items,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            st.session_state[chat_key].append({
                "role":              "assistant",
                "content":           data.get("response_text", "Here are my picks:"),
                "recommendations":   data.get("recommendations", []),
                "meal_plan":         data.get("meal_plan") if show_meal_plan else None,
                "calories_per_meal": data.get("calories_per_meal"),
                "msg_id":            msg_id,
            })
        except Exception as e:
            st.session_state[chat_key].append({
                "role": "assistant",
                "content": f"Something went wrong: {e}",
                "msg_id": msg_id,
            })
    st.rerun()


# ─── Entry point ──────────────────────────────────────────────────────────────

if not st.session_state.onboarded:
    render_onboarding()
else:
    render_main_app()
