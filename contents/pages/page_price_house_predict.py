import streamlit as st
import pandas as pd
st.set_page_config(
    page_title="RealHouse",
    page_icon="📊",
)
st.sidebar.success("Выберете интересующий раздел")
st.title('Оценка стоимости недвижимости')
st.subheader('Выберите параметры квартиры, значения которых вы знаете')
df = pd.DataFrame(columns=['total_area', 'floor', 'floors_number', 'house_age',
       'living_area', 'kitchen_area', 'longitude', 'latitude', 'mean_metro',
       'metro_dist', 'metro_transport_time', 'region', 'rooms_count',
       'is_complete', 'house_material', 'parking', 'passenger_elevator',
       'is_apartments', 'is_auction', 'is_first_floor', 'is_last_floor',
       'has_metro'])
options = st.multiselect("Признаки",['Общая площадь', 'Номер этажа', 'Количество этажей', 'Возраст дома', 'Жилая_площадь', 'Площадь_кухни', 'Долгота_квартиры', 'Широта_квартиры', 'Среднее_расстояние_до_метро', 'Расстояние_до_метро', 'Время_транспортировки_до_метро', 'Регион', 'Количество_комнат', 'Готовность', 'Материал_дома', 'Парковка', 'Пассажирский_лифт', 'Это_апартаменты', 'Аукцион', 'Первый_этаж', 'Последний_этаж', 'Наличие_метро'])

for option in options:
    if option == 'Общая площадь':
        total_area = st.number_input('Площадь дома в кв. метрах', value=None, min_value=5.0, max_value=2000.0, step=0.1,
                                     placeholder='Введите число...')
    elif option == 'Номер этажа':
        floor = st.slider('Номер этажа квартиры', 1, 100)
    elif option == 'Количество этажей':
        floors_number = st.slider('Количество этажей в квартире', 1, 100)
