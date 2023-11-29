from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from typing import List
import pandas as pd
import re
import pickle
import io
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
import logging

LOGGING = 'logs.log'
logging.basicConfig(
    level=logging.DEBUG,
    filename=LOGGING,
    filemode="w"
)


def find_decimal(x) -> str:
    try:
        return re.findall(r'\d+\.?\d+', x)[0]
    except Exception:
        return 'NaN'


with open('onehotencoder.pkl', 'rb') as file:
    onehotencoder = pickle.load(file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:

    '''Предсказание цены для одной машины. На вход подается описание
        машины, функция возвращает одно число - предсказанную стоимость'''

    data = pd.DataFrame(vars(item), index=[0])
    description = pd.read_csv('description.csv', sep=',')
    data['mileage'][data['mileage'].notna()] = data['mileage'][data['mileage'].notna()].apply(find_decimal)
    data['engine'][data['engine'].notna()] = data['engine'][data['engine'].notna()].apply(find_decimal)
    data['max_power'][data['max_power'].notna()] = data['max_power'][data['max_power'].notna()].apply(find_decimal)
    data = data.drop(columns='torque')
    data = data.astype({'mileage': float, 'engine': float, 'max_power': float})
    data['mileage'] = data['mileage'].fillna(description.iloc[5]['mileage'])
    data['engine'] = data['engine'].fillna(description.iloc[5]['engine'])
    data['max_power'] = data['max_power'].fillna(description.iloc[5]['max_power'])
    data['seats'] = data['seats'].fillna(description.iloc[5]['seats'])
    data = data.astype({'engine': int, 'seats': int})
    data_cat = data[['fuel', 'seller_type', 'transmission', 'owner', 'seats']]
    data_cat = data_cat.astype({'seats': str})
    try:
        data_new = onehotencoder.transform(data_cat).toarray()
        data_cat_new = pd.DataFrame(data=data_new, columns=onehotencoder.get_feature_names_out())
    except Exception as error:
        logging.error(error)
    X = data.drop(columns=['name', 'fuel', 'seller_type', 'transmission', 'owner', 'selling_price', 'seats']).join(data_cat_new)
    X['new_1'] = X.year ** 2
    X['new_2'] = X.max_power / X.engine
    X['new_3'] = X.max_power ** 2
    try:
        pred = model.predict(X)
    except Exception as error:
        logging.error(error)
    return pred


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:

    '''Предсказание стоимости для нескольких машин. На вход подаётся список с
        описанием машин, функция возвращает список предсказанных стоимостей машин'''

    data = pd.DataFrame([vars(item) for item in items])
    description = pd.read_csv('description.csv', sep=',')
    data['mileage'][data['mileage'].notna()] = data['mileage'][data['mileage'].notna()].apply(find_decimal)
    data['engine'][data['engine'].notna()] = data['engine'][data['engine'].notna()].apply(find_decimal)
    data['max_power'][data['max_power'].notna()] = data['max_power'][data['max_power'].notna()].apply(find_decimal)
    data = data.drop(columns='torque')
    data = data.astype({'mileage': float, 'engine': float, 'max_power': float})
    data['mileage'] = data['mileage'].fillna(description.iloc[5]['mileage'])
    data['engine'] = data['engine'].fillna(description.iloc[5]['engine'])
    data['max_power'] = data['max_power'].fillna(description.iloc[5]['max_power'])
    data['seats'] = data['seats'].fillna(description.iloc[5]['seats'])
    data = data.astype({'engine': int, 'seats': int})
    data_cat = data[['fuel', 'seller_type', 'transmission', 'owner', 'seats']]
    data_cat = data_cat.astype({'seats': str})
    try:
        data_new = onehotencoder.transform(data_cat).toarray()
        data_cat_new = pd.DataFrame(data=data_new, columns=onehotencoder.get_feature_names_out())
    except Exception as error:
        logging.error(error)
    X = data.drop(columns=['name', 'fuel', 'seller_type', 'transmission', 'owner', 'selling_price', 'seats']).join(
        data_cat_new)
    X['new_1'] = X.year ** 2
    X['new_2'] = X.max_power / X.engine
    X['new_3'] = X.max_power ** 2
    try:
        pred = model.predict(X)
    except Exception as error:
        logging.error(error)
    return pred

@app.post("/predict_items_csv")
def predict_items_csv(file: UploadFile) -> StreamingResponse:

    '''Предсказание стоимости для нескольких машин. На вход подаётся csv-файл с
        описанием машин, функция возвращает csv-файл с ещё одной строкой - предсказанием цен
        автомобилей'''

    content = file.file.read()
    buffer = io.BytesIO(content)
    df = pd.read_csv(buffer, sep=',', index_col=0)
    df = df.astype({'name': str,
                  'year': int, 'selling_price': int,
                  'km_driven': int, 'fuel': str, 'seller_type': str,
                  'transmission': str, 'owner': str,
                  'mileage': str, 'engine': str,
                  'max_power': str, 'torque': str,
                  'seats': float})
    data = df.copy()
    try:
        Items(objects=data.to_dict('records'))
    except ValidationError as error:
        logging.error(error)
    description = pd.read_csv('description.csv', sep=',')
    data['mileage'][data['mileage'].notna()] = data['mileage'][data['mileage'].notna()].apply(find_decimal)
    data['engine'][data['engine'].notna()] = data['engine'][data['engine'].notna()].apply(find_decimal)
    data['max_power'][data['max_power'].notna()] = data['max_power'][data['max_power'].notna()].apply(find_decimal)
    data = data.drop(columns='torque')
    data = data.astype({'mileage': float, 'engine': float, 'max_power': float})
    data['mileage'] = data['mileage'].fillna(description.iloc[5]['mileage'])
    data['engine'] = data['engine'].fillna(description.iloc[5]['engine'])
    data['max_power'] = data['max_power'].fillna(description.iloc[5]['max_power'])
    data['seats'] = data['seats'].fillna(description.iloc[5]['seats'])
    data = data.astype({'engine': int, 'seats': int})
    data_cat = data[['fuel', 'seller_type', 'transmission', 'owner', 'seats']]
    data_cat = data_cat.astype({'seats': str})
    try:
        data_new = onehotencoder.transform(data_cat).toarray()
        data_cat_new = pd.DataFrame(data=data_new, columns=onehotencoder.get_feature_names_out())
    except Exception as error:
        logging.error(error)
    X = data.drop(columns=['name', 'fuel', 'seller_type', 'transmission', 'owner', 'selling_price', 'seats']).join(
        data_cat_new)
    X['new_1'] = X.year ** 2
    X['new_2'] = X.max_power / X.engine
    X['new_3'] = X.max_power ** 2
    try:
        pred = model.predict(X)
    except Exception as error:
        logging.error(error)
    df['price_prediction']=pred
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(
        iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"
    return response
