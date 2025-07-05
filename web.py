import streamlit as st
from streamlit.navigation.page import StreamlitPage

chat_page: StreamlitPage = st.Page(
    page="src/bmserver/chat/page.py",
    title="对话（Chat）大模型",
    icon="📊",
    url_path="chat",
)

page: StreamlitPage = st.navigation(pages=[chat_page])

st.set_page_config(
    page_title="BMServer 推理性能测试数据库",
    layout="wide",
    menu_items={"About": "BMServer 推理性能测试数据库"},
)

page.run()
