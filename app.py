from flask import Flask, request
import joblib
import numpy
import pandas

MODEL_PATH = 'mlmodels/model.pkl'
SCALER_X_PATH = 'mlmodels/scaler_x.pkl'
SCALER_Y_PATH = 'mlmodels/scaler_y.pkl'

app = Flask(__name__)
model = joblib.load(MODEL_PATH)
sc_X = joblib.load(SCALER_X_PATH)
sc_y = joblib.load(SCALER_Y_PATH)

@app.route("/predict_price", methods = ['GET'])
def predict():
    args = request.args
    floor = args.get('floor', default=-1, type=int)
    open_plan = args.get('open_plan', default=-1, type=int)
    rooms = args.get('rooms', default=-1, type=int)
    studio = args.get('studio', default=-1, type=int)
    area = args.get('area', default=-1, type=float)
    kitchen_area = args.get('kitchen_area', default=-1, type=float)
    living_area = args.get('living_area', default=-1, type=float)
    renovation = args.get('renovation', default= -1, type=int)
    building_id = args.get('building_id', default=-1, type=int)
    year = args.get('year', default=-1, type=int)

    x = pandas.DataFrame([{'floor': floor, 'open_plan': open_plan, 'rooms': rooms, 'studio': studio, 'area': area,
                       'kitchen_area': kitchen_area,
                       'living_area': living_area, 'renovation': renovation, 'building_id': building_id, 'year': year,
                       'av_room_area': living_area / rooms}])
    x.area = sc_X.transform(x[['area']])
    x.kitchen_area = sc_X.transform(x[['kitchen_area']])
    x.living_area = sc_X.transform(x[['living_area']])
    x.av_room_area = sc_X.transform(x[['av_room_area']])

    result = sc_y.inverse_transform(model.predict(x).reshape(1,-1))

    return str(result[0][0])


if __name__ == '__main__':
    app.run(debug=True, port=5444, host='0.0.0.0')