from st_pages import Page, show_pages, add_page_title
import  streamlit as st

show_pages(
    [
        Page("contents/pages/home.py", "О проекте", "🏠"),
        Page("contents/pages/page_price_house_predict.py", "Оценка стоимости недвижимости", ":chart_with_upwards_trend:"),
    ]
)