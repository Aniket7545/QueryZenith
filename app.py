import streamlit as st


# --- PAGE SETUP ---
project_2_page = st.Page(
    "src/chatbot.py",
    title="Chat Bot",
    icon=":material/smart_toy:",
    default=True,
)
about = st.Page(
    "src/about.py",
    title="About us",
    icon=":material/account_circle:",
    
)
project_1_page = st.Page(
    "src/data.py",
    title="Sales Dashboard",
    icon=":material/bar_chart:",
)


# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
# pg = st.navigation(pages=[about_page, project_1_page, project_2_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Main Product": [project_2_page],
        "Data Analysis": [project_1_page],
        "About":[about],
    }
)


# # --- SHARED ON ALL PAGES ---
st.logo("assets/s1.png")
# st.sidebar.markdown("Made with ❤️ by Sanyam Sankhala")


# --- RUN NAVIGATION ---
pg.run()
