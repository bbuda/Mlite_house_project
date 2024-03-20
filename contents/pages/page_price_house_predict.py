import streamlit as st
import pandas as pd
st.set_page_config(
    page_title="RealHouse",
    page_icon="📊",
)
st.sidebar.success("Выберете интересующий раздел")
st.title('Оценка стоимости недвижимости')
st.subheader('Выберите параметры квартиры, значения которых вы знаете')
df = pd.DataFrame(columns=['region', 'address', 'total_area', 'kitchen_area',
       'living_area', 'rooms_count', 'floor', 'floors_number', 'build_date',
       'is_complete', 'completion_year', 'house_material', 'parking',
       'decoration', 'balcony', 'longitude', 'latitude', 'passenger_elevator',
       'cargo_elevator', 'metro', 'metro_distance', 'metro_transport',
       'district', 'is_apartments', 'is_auction'])
options = st.multiselect("Признаки",['Регион', 'Адрес', 'Общая площадь', 'Площадь кухни', 'Жилая площадь', 'Количество комнат',
                                     'Номер этажа', 'Количество этажей', 'Год начала строительства дома', 'Дом построен',
                                     'Год завершения строительства дома', 'Материал дома', 'Тип парковки', 'Тип отделки',
                                     'Число балконов', 'Долгота дома', 'Широта дома', 'Число пассажирских лифтов',
                                     'Число грузовых лифтов', 'Метро', 'Район', 'Квартира - апартаменты', 'Квартира на аукционе'])

for option in options:
    if option == 'Регион':
        region = st.radio('Область расположения квартиры:', ['spb', 'msk',  'kzn',  'nng', 'ekb', 'nsk'])
    elif option == 'Адрес':
        address = st.text_area('Адрес дома:')
    elif option == 'Общая площадь':
        total_area = st.number_input('Площадь дома в кв. метрах:', value=None, min_value=5.0, max_value=2000.0, step=0.1,
                                     placeholder='Введите число...')
    elif option == 'Площадь кухни':
        total_area = st.number_input('Площадь кухни в кв. метрах:', value=None, min_value=5.0, max_value=2000.0, step=0.1,
                                     placeholder='Введите число...')
    elif option == 'Жилая площадь':
        total_area = st.number_input('Жилая площадь в кв. метрах:', value=None, min_value=5.0, max_value=2000.0, step=0.1,
                                     placeholder='Введите число...')
    elif option == 'Количество комнат':
        rooms_count = st.radio('Количество комнат в квартире:', [i for i in range(1, 9)])
    elif option == 'Номер этажа':
        floor = st.slider('Номер этажа квартиры', 1, 100)
    elif option == 'Количество этажей':
        floors_number = st.slider('Количество этажей в квартире', 1, 100)
    elif option == 'Год начала строительства дома':
        build_date = st.number_input('Год начала строительства дома:', value=None, min_value=1800, max_value=2024, step=1,
                                     placeholder='Введите год...')
    elif option == 'Дом построен':
        is_complete= st.radio('Строительство дома завершено?', ['Да', 'Нет'])

    elif option == 'Год завершения строительства дома':
        completion_year = st.number_input('Год завершения строительства дома:', value=None, min_value=1860, max_value=2029,
                                     step=1,
                                     placeholder='Введите год...')
    elif option == 'Основной материал дома':
        house_material = st.radio('Основной материал из которого сделан дом:', ['Монолитный кирпич', 'Монолит', 'Кирпич',
                                                                                'Панель', 'Сталь', 'Старый материал', 'Дерево',
                                                                                'Газосиликатный блок'])
    elif option == 'Тип парковки':
        parking = st.selectbox('Тип парковки у дома:', ['Подземная', 'Наземная', 'Мультиуровневая', 'Открытая'])
    elif option == 'Тип отделки':
        decoration = st.selectbox('Тип отделки квартиры:', ['Без отделки', 'Грубая отделка', 'Небольшая отделка'])
    elif option == 'Число балконов':
        balcony = st.selectbox('Число балконов в квартире:', [0, 1, 2, 3, 4, 5])
    elif option == 'Долгота дома':
        longitude = st.number_input('Коордианта долготы дома:', value=None, min_value=29.04, max_value=84.01, step=0.01)
    elif option == 'Широта дома':
        latitude = st.number_input('Коордианта широты дома:', value=None, min_value=54.85, max_value=61.00, step=0.01)
    elif option == 'Число пассажирских лифтов':
        passenger_elevator = st.number_input('Число пассажирских лифтов в доме', value=2, min_value=0, max_value=35, step=1)
    elif option == 'Число грузовых лифтов':
        cargo_elevator = st.number_input('Число грузовых лифтов в доме', value=2, min_value=0, max_value=35, step=1)
    elif option == 'Метро':
        metro = st.text_area('Ближайшие станции от дома к метро через запятую:')
        st.write(metro)
        for m in (metro.split(',')):
            metro_distance = st.slider(f'Расстояние в км от дома до метро {m}', 1, 60)
            metro_transport = st.radio(f'Тип перемещения от дома до метро {m}', ["Пешком","На транспорте"])
    elif option == 'Район':
        district = st.text_area('Район города, в котором располагается дом:')
    elif option == 'Квартира - апартаменты':
        is_apartments = st.checkbox('Квартира является апартаментами')
    elif option == 'Квартира на аукционе':
        is_auction = st.checkbox('Квартира выставлена на аукцион?')






