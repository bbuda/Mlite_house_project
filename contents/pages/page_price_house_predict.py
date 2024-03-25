import pickle

import joblib
import streamlit as st
import pandas as pd
import catboost
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

st.set_page_config(
    page_title="RealHouse",
    page_icon="📊",
)
st.sidebar.success("Выберете интересующий раздел")
st.title('Оценка стоимости недвижимости')
st.subheader('Выберите параметры квартиры, значения которых вы знаете')
st.text(''' Для более точного прогноза, пожалуйса, заполните поля с количеством комнат в квартире, 
общей площадью, регионом расположения квартиры.
Поле "Дом построен" обзательно для заполнения'''
        )
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
            i += str(metro_distance) + ', ' * (m != metro.split((','))[-1])
            metro_transport = st.radio(f'Тип перемещения от дома до метро {m}', ["Пешком","На транспорте"])
            ii += str(metro_transport) + ", " * (m != metro.split((','))[-1])
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
        lambda x: 'Панель' if pd.isna(x.house_material) and x.floors_number in (5, 9) else x.house_material, axis=1)

    # Замена значений 'stalin' на 'brick' в столбце 'house_material'
    # Замена значений в столбце house_material
    df['house_material'] = df['house_material'].replace(['Сталь', 'Газосиликатный блок'], 'Кирпич')
    df['house_material'] = df['house_material'].replace(['Дерево', 'Старый материал'], 'Панель')



    def calculate_passenger_elevator(x):
        if pd.isna(x.passenger_elevator):
            if x.floors_number in range(1, 6) and x.house_material == 'Панель':
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
    df['parking'] = df.apply(lambda x: 'Открытая' if pd.isna(x.parking) else x.parking, axis=1)
    # Вычисление возраста дома
    df['house_age'] = 2024 - df['build_date']

    # Обновление значений 'house_age' в зависимости от статуса завершенности дома
    df.loc[(df['is_complete'] == 0) & (df['house_age'].isna()), 'house_age'] = 0
    df.loc[(df['is_complete'] == 1) & (df['house_age'].isna()), 'house_age'] = 2024 - df['completion_year']
    df['is_first_floor'] = df.apply(lambda x: 1 if x.floor == 1 else 0, axis=1)
    df['is_last_floor'] = df.apply(lambda x: 1 if x.floor == x.floors_number else 0, axis=1)
    df['has_metro'] = df.apply(lambda x: 0 if pd.isna(x.metro) else 1, axis=1)


    def calculate_mean_metro(row):
        if row['has_metro'] == 0:
            return 0
        metro_distances = row['metro_distance'].split(',')
        metro_distances = [float(distance) for distance in metro_distances if
                           distance.strip()]  # Исключаем пустые строки
        if not metro_distances:
            return 0
        return np.mean(metro_distances)


    # Применяем функцию к DataFrame
    df['mean_metro'] = df.apply(calculate_mean_metro, axis=1)
    st.write(df)
    d = {'total_area': 61.1,
         'floor': 5.0,
         'floors_number': 15.0,
         'house_age': 0.0,
         'living_area': 30.0,
         'kitchen_area': 14.6,
         'mean_metro': 7.0}
    df.fillna(value=d, inplace=True)
    st.write(df)
    dd = {'region': 'msk',
         'rooms_count': 2.0,
         'is_complete': 1.0,
         'house_material': 'Монолит',
         'parking': 'Открытая',
         'passenger_elevator': 1.0,
         'cargo_elevator': 1.0,
         'is_apartments': 0.0,
         'is_auction': 0,
         'is_first_floor': 0,
         'is_last_floor': 0,
         'has_metro': 1}
    df.fillna(value=dd, inplace=True)
    num_f = ['total_area',
             'floor',
             'floors_number',
             'house_age',
             'living_area',
             'kitchen_area',
             'longitude',
             'latitude',
             'mean_metro',
             ]
    cat_f = ['region',
             #          'decoration',
             'rooms_count',
             'is_complete',
             'house_material',
             'parking',
             'passenger_elevator',
             'cargo_elevator',
             'is_apartments',
             'is_auction',
             'is_first_floor',
             'is_last_floor',
             'has_metro',
             ]

    df = df[num_f + cat_f].copy()
    st.write(df)
    # Заменяем категориальные переменные числами
    df.replace({'region': {'msk': 2, 'spb': 5, 'ekb': 0, 'nsk' : 4, 'nng' : 3, 'kzn' : 1 },
                'house_material': {'Монолитный кирпич': 1, 'Монолит': 2, 'Кирпич': 0, 'Панель': 3},
                'parking': {'Открытая': 2, 'Подземная': 3, 'Наземная' : 0,  'Мультиуровневая': 1},
                # Другие категориальные переменные и их соответствия числам
                }, inplace=True)
    st.write(df)
    st.write(df.dtypes)

    if df.iloc[0]['rooms_count'] == 1:
        model = joblib.load(open('models_configs/model_(1)rc.joblib', 'rb'))
        price_predict = model.predict(df)
        st.subheader(f'Стоимость этой квартиры оценивается: {round(price_predict, 2)} рублей')
        st.write(price_predict)
    if df.iloc[0]['rooms_count'] == 2:
        model = joblib.load(open('models_configs/model_(2)rc.joblib', 'rb'))
        price_predict = model.predict(df)
        st.subheader(f'Стоимость этой квартиры оценивается: {round(price_predict, 2)} рублей')
    if df.iloc[0]['rooms_count'] == 3:
        model = joblib.load(open('models_configs/model_(2)rc.joblib', 'rb'))
        price_predict = model.predict(df)
        st.subheader(f'Стоимость этой квартиры оценивается: {round(price_predict, 2)} рублей')
    if df.iloc[0]['rooms_count'] == 4:
        model = joblib.load(open('models_configs/model_(2)rc.joblib', 'rb'))
        price_predict = model.predict(df)
        st.subheader(f'Стоимость этой квартиры оценивается: {round(price_predict, 2)} рублей')
    if df.iloc[0]['rooms_count'] > 4:
        model = joblib.load(open('models_configs/model_(2)rc.joblib', 'rb'))
        price_predict = model.predict(df)
        st.subheader(f'Стоимость этой квартиры оценивается: {round(price_predict, 2)} рублей')
    import plotly.express as px


    def plot_with_prediction_highlight(df, prediction_df):
        fig = px.scatter(df, x='total_area', y='price', color_discrete_sequence=['blue'], title='Общая площадь и цена')
        fig.add_scatter(x=prediction_df['total_area'], y=prediction_df['price_pred'], mode='markers',
                        marker=dict(color='red', size=10), name='Предсказание')
        fig.update_xaxes(title_text='Общая площадь (кв.м)')
        fig.update_yaxes(title_text='Цена, руб.')
        fig.show()


    def plot_dist_with_prediction(df, prediction_df):
        fig = px.histogram(df, x='price', color_discrete_sequence=['blue'], marginal='rug',
                           title='Распределение цены и предсказанная цена')

        for _, row in prediction_df.iterrows():
            fig.add_vline(x=row['price_pred'], line_dash='dash', line_color='red', annotation_text='Предсказанная цена')

        fig.update_layout(xaxis_title='Цена, руб.', yaxis_title='Плотность')
        fig.show()









