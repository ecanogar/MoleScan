import streamlit as st

pg = st.navigation([
    st.Page("page1.py", title="Calcula número de lunares", icon="🔎"),
    st.Page("page2.py", title="Dimensiones del lunar", icon="📏"),
    st.Page("page3.py", title="Clasificación dermatoscópica", icon="👩‍⚕️"),
    
])
pg.run()