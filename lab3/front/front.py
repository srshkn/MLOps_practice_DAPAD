import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")

st.title("Lab 3")

if st.button("Проверить API"):
    r = requests.get(f"{API_URL}/_info", timeout=10)
    st.json(r.json())

if st.button("Загрузить датасет"):
    r = requests.post(f"{API_URL}/load-data", json={}, timeout=60)
    st.json(r.json())

if st.button("Preprocess"):
    r = requests.post(f"{API_URL}/preprocess", timeout=60)
    st.json(r.json())