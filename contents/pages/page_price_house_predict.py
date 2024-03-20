import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
st.set_page_config(
    page_title="RealHouse",
    page_icon="📊",
)
st.sidebar.success("Выберете интересующий раздел")
st.title('Оценка стоимости недвижимости')
st.subheader('Выберите параметры квартиры, значения которых вы знаете')
c = ['region', 'address', 'total_area', 'kitchen_area',
       'living_area', 'rooms_count', 'floor', 'floors_number', 'build_date',
       'is_complete', 'completion_year', 'house_material', 'parking',
       'decoration', 'balcony', 'longitude', 'latitude', 'passenger_elevator',
       'cargo_elevator', 'metro', 'metro_distance', 'metro_transport',
       'district', 'is_apartments', 'is_auction']
df = pd.DataFrame(columns=c)

data = {}
for i in c:
    data[i] = np.nan
options = st.multiselect("Признаки",['Регион', 'Адрес', 'Общая площадь', 'Площадь кухни', 'Жилая площадь', 'Количество комнат',
                                     'Номер этажа', 'Количество этажей', 'Год начала строительства дома', 'Дом построен',
                                     'Год завершения строительства дома', 'Основной материал дома', 'Тип парковки', 'Тип отделки',
                                     'Число балконов', 'Долгота дома', 'Широта дома', 'Число пассажирских лифтов',
                                     'Число грузовых лифтов', 'Метро', 'Район', 'Квартира - апартаменты', 'Квартира на аукционе'])
flag = st.checkbox('Все известные параметры квартиры собраны?')
for option in options:
    if option == 'Регион':
        region = st.radio('Область расположения квартиры:', ['spb', 'msk',  'kzn',  'nng', 'ekb', 'nsk'])
        data['region'] = region
    elif option == 'Адрес':
        address = st.text_area('Адрес дома:')
        data['address'] = address
    elif option == 'Общая площадь':
        total_area = st.number_input('Площадь дома в кв. метрах:', value=None, min_value=5.0, max_value=2000.0, step=0.1,
                                     placeholder='Введите число...')
        data['total_area'] = total_area
    elif option == 'Площадь кухни':
        kitchen_area = st.number_input('Площадь кухни в кв. метрах:', value=None, min_value=5.0, max_value=2000.0, step=0.1,
                                     placeholder='Введите число...')
        data['kitchen_area'] = kitchen_area
    elif option == 'Жилая площадь':
        living_area = st.number_input('Жилая площадь в кв. метрах:', value=None, min_value=5.0, max_value=2000.0, step=0.1,
                                     placeholder='Введите число...')
        data['living_area'] = living_area
    elif option == 'Количество комнат':
        rooms_count = st.radio('Количество комнат в квартире:', [i for i in range(1, 9)])
        data['rooms_count'] = rooms_count
    elif option == 'Номер этажа':
        floor = st.slider('Номер этажа квартиры', 1, 100)
        data['floor'] = floor
    elif option == 'Количество этажей':
        floors_number = st.slider('Количество этажей в квартире', 1, 100)
        data['floors_number'] = floors_number
    elif option == 'Год начала строительства дома':
        build_date = st.number_input('Год начала строительства дома:', value=None, min_value=1800, max_value=2024, step=1,
                                     placeholder='Введите год...')
        data['build_date'] = build_date
    elif option == 'Дом построен':
        is_complete= st.radio('Строительство дома завершено?', ['Да', 'Нет'])
        data['is_complete'] = is_complete

    elif option == 'Год завершения строительства дома':
        completion_year = st.number_input('Год завершения строительства дома:', value=None, min_value=1860, max_value=2029,
                                     step=1,
                                     placeholder='Введите год...')
        data['completion_year'] = completion_year
    elif option == 'Основной материал дома':
        house_material = st.radio('Основной материал из которого сделан дом:', ['Монолитный кирпич', 'Монолит', 'Кирпич',
                                                                                'Панель', 'Сталь', 'Старый материал', 'Дерево',
                                                                                'Газосиликатный блок'])
        data['house_material'] = house_material
    elif option == 'Тип парковки':
        parking = st.selectbox('Тип парковки у дома:', ['Подземная', 'Наземная', 'Мультиуровневая', 'Открытая'])
        data['parking'] = parking
    elif option == 'Тип отделки':
        decoration = st.selectbox('Тип отделки квартиры:', ['Без отделки', 'Грубая отделка', 'Небольшая отделка'])
        data['decoration'] = decoration
    elif option == 'Число балконов':
        balcony = st.selectbox('Число балконов в квартире:', [0, 1, 2, 3, 4, 5])
        data['balcony'] = balcony
    elif option == 'Долгота дома':
        longitude = st.number_input('Коордианта долготы дома:', value=None, min_value=29.04, max_value=84.01, step=0.01)
        data['longitude'] = longitude
    elif option == 'Широта дома':
        latitude = st.number_input('Коордианта широты дома:', value=None, min_value=54.85, max_value=61.00, step=0.01)
        data['latitude'] = latitude
    elif option == 'Число пассажирских лифтов':
        passenger_elevator = st.number_input('Число пассажирских лифтов в доме', value=2, min_value=0, max_value=35, step=1)
        data['passenger_elevator'] = passenger_elevator
    elif option == 'Число грузовых лифтов':
        cargo_elevator = st.number_input('Число грузовых лифтов в доме', value=2, min_value=0, max_value=35, step=1)
        data['cargo_elevator'] = cargo_elevator
    elif option == 'Метро':
        metro = st.text_area('Ближайшие станции от дома к метро через запятую:')
        i, ii = '', ''
        data['metro'] = metro
        for m in (metro.split(',')):
            metro_distance = st.slider(f'Расстояние в км от дома до метро {m}', 1, 60)
            i += str(metro_distance) + ', '
            metro_transport = st.radio(f'Тип перемещения от дома до метро {m}', ["Пешком","На транспорте"])
            ii += str(metro_transport) + ", "
        data['metro_distance'] = i
        data['metro_transport'] = ii
    elif option == 'Район':
        district = st.text_area('Район города, в котором располагается дом:')
        data['district'] = district
    elif option == 'Квартира - апартаменты':
        is_apartments = st.checkbox('Квартира является апартаментами')
        data['is_apartments'] = is_apartments
    elif option == 'Квартира на аукционе':
        is_auction = st.checkbox('Квартира выставлена на аукцион?')
        data['is_auction'] = is_auction
if flag:
    data_lists = {key: [value] for key, value in data.items()}
    df = pd.DataFrame.from_dict(data_lists)
    st.write(df)
    df['is_complete'] = df.apply(lambda x: int(x.completion_year < 2024) if pd.isna(x.is_complete) and not pd.isna(x.completion_year) else x.is_complete, axis=1)

    # Второй блок кода
    df['is_complete'] = df.apply(lambda x: int(x.build_date < 2024) if pd.isna(x.is_complete) and not pd.isna(x.build_date) else x.is_complete, axis=1)

    # Третий блок кода
    df['is_complete'] = df.apply(lambda x: int(x.is_auction) if pd.isna(x.is_complete) else x.is_complete, axis=1)
    st.write(df)
    df['decoration'] = df.apply(lambda x: 'without' if pd.isna(x.decoration) and not x.is_complete else x.decoration,
                                axis=1)
    df['house_material'] = df.apply(
        lambda x: 'panel' if pd.isna(x.house_material) and x.floors_number in (5, 9) else x.house_material, axis=1)

    # Замена значений 'stalin' на 'brick' в столбце 'house_material'
    df.loc[df['house_material'] == 'stalin', 'house_material'] = 'brick'



    def calculate_passenger_elevator(x):
        if pd.isna(x.passenger_elevator):
            if x.floors_number in range(1, 6) and x.house_material == 'panel':
                return 0
            elif x.floors_number == 9:
                return 1
            elif 10 <= x.floors_number <= 19:
                return 2
            elif 20 <= x.floors_number <= 25:
                return 3
            elif x.floors_number >= 26:
                return 4
        return x.passenger_elevator


    # Применяем функцию к DataFrame
    df['passenger_elevator'] = df.apply(calculate_passenger_elevator, axis=1)


    def calculate_cargo_elevator(x):
        if pd.isna(x.cargo_elevator):
            if x.floors_number in range(1, 10):
                return 0
            elif 10 <= x.floors_number <= 12 and x.passenger_elevator == 0:
                return 1
            elif 12 <= x.floors_number <= 22 and x.passenger_elevator in (1, 2):
                return 1
            elif 23 <= x.floors_number <= 29 and x.passenger_elevator in (1, 2):
                return 2
        return x.cargo_elevator


    # Применяем функцию к DataFrame
    df['cargo_elevator'] = df.apply(calculate_cargo_elevator, axis=1)
    df['parking'] = df.apply(lambda x: 'open' if pd.isna(x.parking) else x.parking, axis=1)
    # Вычисление возраста дома
    df['house_age'] = 2024 - df['build_date']

    # Обновление значений 'house_age' в зависимости от статуса завершенности дома
    df.loc[(df['is_complete'] == 0) & (df['house_age'].isna()), 'house_age'] = 0
    df.loc[(df['is_complete'] == 1) & (df['house_age'].isna()), 'house_age'] = 2024 - df['completion_year']
    df['first_floor'] = df.apply(lambda x: 1 if x.floor == 1 else 0, axis=1)
    df['last_floor'] = df.apply(lambda x: 1 if x.floor == x.floors_number else 0, axis=1)
    df['has_metro'] = df.apply(lambda x: 0 if pd.isna(x.metro) else 1, axis=1)
    df['mean_metro'] = df.apply(lambda x: 0 if x.has_metro == 0 else sum(map(float, x.metro_distance.split(','))) / len(x.metro.split(',')), axis=1)


    def calculate_mean_distance(row):
        distances = [int(x) for x in str(row['metro_distance']).split(',') if x.isdigit()]
        transports = str(row['metro_transport']).split(',')

        total_distance = 0
        total_transport_time = 0

        for i in range(len(distances)):
            if transports[i] == 'walk':
                total_distance += distances[i] / 60 * 5
            elif transports[i] == 'transport':
                total_distance += distances[i] / 60 * 40
                total_transport_time += (distances[i] / 60)

        if len(distances) > 0:
            return total_distance / len(distances), total_transport_time
        else:
            return None, None


    df[['metro_dist', 'metro_transport_time']] = df.apply(calculate_mean_distance, axis=1, result_type='expand')
    st.write(df)

