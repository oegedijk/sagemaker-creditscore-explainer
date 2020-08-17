from pathlib import Path
import joblib
from flask import Flask, request, abort, jsonify
import json

import numpy as np
import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
transformer = joblib.load(open(str(Path.cwd() / "pkl" / "transformer.joblib"), "rb"))
model = joblib.load(open(str(Path.cwd() / "pkl" / "model.joblib"), "rb"))
shap_explainer = joblib.load(open(str(Path.cwd() / "pkl" / "shap_explainer.joblib"), "rb"))

flask_app = Flask(__name__)
positive_label = 1
shap_base = shap_explainer.expected_value[1]

@flask_app.route('/predict', methods=['Post'])
def prediction_server_API_call():
    try:
        json_str = request.get_json()
        df = pd.read_json(json_str, orient='records')
    except Exception as e:
        raise e

    if df.empty:
        return(abort(400))
    else:
        features = transformer.transform(df)
        shap_values = np.transpose(shap_explainer.shap_values(features)[positive_label])
        preds = model.predict_proba(features)[:, positive_label]
        response = {}
        response['prediction'] = preds
        response['shap_base'] = shap_base
        response['shap_values'] = {k: v for k, v in zip(features.columns.tolist(), shap_values.tolist())}
        return json.dumps(response, cls=NumpyEncoder)


if __name__=="__main__":
    flask_app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)