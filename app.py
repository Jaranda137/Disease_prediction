from flask import Flask
from flask import Blueprint, render_template, send_from_directory, request
import json
import joblib
import onnxruntime as rt
import numpy as np

app = Flask(__name__)


@app.route("/")
def home():
  return render_template("index.html")


@app.route('/predict', methods=['POST'])
def guardar_datos():
  enfermedad = request.form['enfermedad']
  algoritmo = request.form['algoritmo']
  model_lg = joblib.load('models/model_lg.joblib')
  model_dt = joblib.load('models/model_dt.joblib')
  model_knn = joblib.load('models/model_knn.joblib')
  model_svm = joblib.load('models/model_svm.joblib')
  model_nb = joblib.load('models/model_nb.joblib')
  model_gb = joblib.load('models/model_gb.joblib')
  model_rf = joblib.load('models/model_rf.joblib')
  #model_ann = joblib.load('models/model_ann.joblib')
  model_lg_diabetes = joblib.load('models/model_lg_diabetes.joblib')
  model_dt_diabetes = joblib.load('models/model_dt_diabetes.joblib')
  model_knn_diabetes = joblib.load('models/model_knn_diabetes.joblib')
  model_svm_diabetes = joblib.load('models/model_svm_diabetes.joblib')
  model_nb_diabetes = joblib.load('models/model_nb_diabetes.joblib')
  model_gb_diabetes = joblib.load('models/model_gb_diabetes.joblib')
  model_rf_diabetes = joblib.load('models/model_rf_diabetes.joblib')
  #model_ann_db = joblib.load('models/model_ann_db.joblib')
  model_lg_hep = joblib.load('models/model_lg_hep.joblib')
  model_dt_hep = joblib.load('models/model_dt_hep.joblib')
  model_knn_hep = joblib.load('models/model_knn_hep.joblib')
  model_svm_hep = joblib.load('models/model_svm_hep.joblib')
  model_nb_hep = joblib.load('models/model_nb_hep.joblib')
  model_gb_hep = joblib.load('models/model_gb_hep.joblib')
  model_rf_hep = joblib.load('models/model_rf_hep.joblib')
  #model_ann_hep = joblib.load('models/model_ann_hep.joblib')
  model_lg_par = joblib.load('models/model_lg_par.joblib')
  model_dt_par = joblib.load('models/model_dt_par.joblib')
  model_knn_par = joblib.load('models/model_knn_par.joblib')
  model_svm_par = joblib.load('models/model_svm_par.joblib')
  model_nb_par = joblib.load('models/model_nb_par.joblib')
  model_gb_par = joblib.load('models/model_gb_par.joblib')
  model_rf_par = joblib.load('models/model_rf_par.joblib')
  #model_ann_par = joblib.load('models/model_ann_par.joblib')
  model_lg_hcc = joblib.load('models/model_lg_hcc.joblib')
  model_dt_hcc = joblib.load('models/model_dt_hcc.joblib')
  model_knn_hcc = joblib.load('models/model_knn_hcc.joblib')
  model_svm_hcc = joblib.load('models/model_svm_hcc.joblib')
  model_nb_hcc = joblib.load('models/model_nb_hcc.joblib')
  model_gb_hcc = joblib.load('models/model_gb_hcc.joblib')
  model_rf_hcc = joblib.load('models/model_rf_hcc.joblib')
  #model_ann_hcc = joblib.load('models/model_ann_hcc.joblib')
  models = {
    'model_lg': model_lg,
    'model_dt': model_dt,
    'model_knn': model_knn,
    'model_svm': model_svm,
    'model_nb': model_nb,
    'model_gb': model_gb,
    'model_rf': model_rf,
    #'model_ann': model_ann,
    'model_lg_diabetes': model_lg_diabetes,
    'model_dt_diabetes': model_dt_diabetes,
    'model_knn_diabetes': model_knn_diabetes,
    'model_svm_diabetes': model_svm_diabetes,
    'model_nb_diabetes': model_nb_diabetes,
    'model_gb_diabetes': model_gb_diabetes,
    'model_rf_diabetes': model_rf_diabetes,
    #'model_ann_db': model_ann_db,
    'model_lg_hep': model_lg_hep,
    'model_dt_hep': model_dt_hep,
    'model_knn_hep': model_knn_hep,
    'model_svm_hep': model_svm_hep,
    'model_nb_hep': model_nb_hep,
    'model_gb_hep': model_gb_hep,
    'model_rf_hep': model_rf_hep,
    #'model_ann_hep': model_ann_hep,
    'model_lg_par': model_lg_par,
    'model_dt_par': model_dt_par,
    'model_knn_par': model_knn_par,
    'model_svm_par': model_svm_par,
    'model_nb_par': model_nb_par,
    'model_gb_par': model_gb_par,
    'model_rf_par': model_rf_par,
    #'model_ann_par': model_ann_par,
    'model_lg_hcc': model_lg_hcc,
    'model_dt_hcc': model_dt_hcc,
    'model_knn_hcc': model_knn_hcc,
    'model_svm_hcc': model_svm_hcc,
    'model_nb_hcc': model_nb_hcc,
    'model_gb_hcc': model_gb_hcc,
    'model_rf_hcc': model_rf_hcc,
    #'model_ann_hcc': model_ann_hcc
  }
  if algoritmo == 'algoritmo1' and enfermedad == 'breast_cancer':
    modelo_seleccionado = models['model_lg']
  elif algoritmo == 'algoritmo2' and enfermedad == 'breast_cancer':
    modelo_seleccionado = models['model_dt']
  elif algoritmo == 'algoritmo3' and enfermedad == 'breast_cancer':
    modelo_seleccionado = models['model_knn']
  elif algoritmo == 'algoritmo4' and enfermedad == 'breast_cancer':
    modelo_seleccionado = models['model_svm']
  elif algoritmo == 'algoritmo5' and enfermedad == 'breast_cancer':
    modelo_seleccionado = models['model_nb']
  elif algoritmo == 'algoritmo6' and enfermedad == 'breast_cancer':
    modelo_seleccionado = models['model_gb']
  elif algoritmo == 'algoritmo7' and enfermedad == 'breast_cancer':
    modelo_seleccionado = models['model_rf']
  elif algoritmo == 'algoritmo8' and enfermedad == 'breast_cancer':
    #modelo_seleccionado = models['model_ann']
    pass
  elif algoritmo == 'algoritmo1' and enfermedad == 'diabetes':
    modelo_seleccionado = models['model_lg_diabetes']
  elif algoritmo == 'algoritmo2' and enfermedad == 'diabetes':
    modelo_seleccionado = models['model_dt_diabetes']
  elif algoritmo == 'algoritmo3' and enfermedad == 'diabetes':
    modelo_seleccionado = models['model_knn_diabetes']
  elif algoritmo == 'algoritmo4' and enfermedad == 'diabetes':
    modelo_seleccionado = models['model_svm_diabetes']
  elif algoritmo == 'algoritmo5' and enfermedad == 'diabetes':
    modelo_seleccionado = models['model_nb_diabetes']
  elif algoritmo == 'algoritmo6' and enfermedad == 'diabetes':
    modelo_seleccionado = models['model_gb_diabetes']
  elif algoritmo == 'algoritmo7' and enfermedad == 'diabetes':
    modelo_seleccionado = models['model_rf_diabetes']
  elif algoritmo == 'algoritmo8' and enfermedad == 'diabetes':
    #modelo_seleccionado = models['model_ann_db']
    pass
  elif algoritmo == 'algoritmo1' and enfermedad == 'hepatitis':
    modelo_seleccionado = models['model_lg_hep']
  elif algoritmo == 'algoritmo2' and enfermedad == 'hepatitis':
    modelo_seleccionado = models['model_dt_hep']
  elif algoritmo == 'algoritmo3' and enfermedad == 'hepatitis':
    modelo_seleccionado = models['model_knn_hep']
  elif algoritmo == 'algoritmo4' and enfermedad == 'hepatitis':
    modelo_seleccionado = models['model_svm_hep']
  elif algoritmo == 'algoritmo5' and enfermedad == 'hepatitis':
    modelo_seleccionado = models['model_nb_hep']
  elif algoritmo == 'algoritmo6' and enfermedad == 'hepatitis':
    modelo_seleccionado = models['model_gb_hep']
  elif algoritmo == 'algoritmo7' and enfermedad == 'hepatitis':
    modelo_seleccionado = models['model_rf_hep']
  elif algoritmo == 'algoritmo8' and enfermedad == 'hepatitis':
    #modelo_seleccionado = models['model_ann_hep']
    pass
  elif algoritmo == 'algoritmo1' and enfermedad == 'parkinson':
    modelo_seleccionado = models['model_lg_par']
  elif algoritmo == 'algoritmo2' and enfermedad == 'parkinson':
    modelo_seleccionado = models['model_dt_par']
  elif algoritmo == 'algoritmo3' and enfermedad == 'parkinson':
    modelo_seleccionado = models['model_knn_par']
  elif algoritmo == 'algoritmo4' and enfermedad == 'parkinson':
    modelo_seleccionado = models['model_svm_par']
  elif algoritmo == 'algoritmo5' and enfermedad == 'parkinson':
    modelo_seleccionado = models['model_nb_par']
  elif algoritmo == 'algoritmo6' and enfermedad == 'parkinson':
    modelo_seleccionado = models['model_gb_par']
  elif algoritmo == 'algoritmo7' and enfermedad == 'parkinson':
    modelo_seleccionado = models['model_rf_par']
  elif algoritmo == 'algoritmo8' and enfermedad == 'parkinson':
    #modelo_seleccionado = models['model_ann_par']
    pass
  elif algoritmo == 'algoritmo1' and enfermedad == 'HCC':
    modelo_seleccionado = models['model_lg_hcc']
  elif algoritmo == 'algoritmo2' and enfermedad == 'HCC':
    modelo_seleccionado = models['model_dt_hcc']
  elif algoritmo == 'algoritmo3' and enfermedad == 'HCC':
    modelo_seleccionado = models['model_knn_hcc']
  elif algoritmo == 'algoritmo4' and enfermedad == 'HCC':
    modelo_seleccionado = models['model_svm_hcc']
  elif algoritmo == 'algoritmo5' and enfermedad == 'HCC':
    modelo_seleccionado = models['model_nb_hcc']
  elif algoritmo == 'algoritmo6' and enfermedad == 'HCC':
    modelo_seleccionado = models['model_gb_hcc']
  elif algoritmo == 'algoritmo7' and enfermedad == 'HCC':
    modelo_seleccionado = models['model_rf_hcc']
  elif algoritmo == 'algoritmo8' and enfermedad == 'HCC':
    #modelo_seleccionado = models['model_ann_hcc']
    pass
  else:
    modelo_seleccionado = None

  data = request.form.to_dict()
  valores = []
  for clave, valor in data.items():
    if clave.startswith('valor') and valor != '':
      try:
        valor_numerico = float(valor)
        valores.append(valor_numerico)
      except ValueError as e:
        valor_numerico = None
        valores = None
        break

  if algoritmo == 'algoritmo8':
    modelo_seleccionado = None
    valores = np.array(valores).astype(np.float32).reshape(1, -1)
    if enfermedad == 'breast_cancer':
      model_ann = rt.InferenceSession('models/model_ann.onnx')
    elif enfermedad == 'diabetes':
      model_ann = rt.InferenceSession('models/model_ann_db.onnx')
    elif enfermedad == 'hepatitis':
      model_ann = rt.InferenceSession('models/model_ann_hep.onnx')
    elif enfermedad == 'parkinson':
      model_ann = rt.InferenceSession('models/model_ann_par.onnx')
    elif enfermedad == 'HCC':
      model_ann = rt.InferenceSession('models/model_ann_hcc.onnx')
    try:
      inputs = model_ann.get_inputs()
      outputs = model_ann.get_outputs()

      # Crear el feed dict con los valores de entrada
      feed_dict = {inputs[0].name: valores}

      # Realizar la predicción
      output = model_ann.run([output.name for output in outputs], feed_dict)

      # Obtener los resultados
      predictions = output[0][0]
      result = [int(predictions[0].round())]
    except Exception as e:
      output = None
      predictions = None
      modelo_seleccionado = None
      result = None

  try:
    if algoritmo == 'algoritmo9' and enfermedad == 'breast_cancer':
      try:
        valores = np.array(valores).astype(np.float32).reshape(1, -1)
        model_ann = rt.InferenceSession('models/model_ann.onnx')
        inputs = model_ann.get_inputs()
        outputs = model_ann.get_outputs()
        feed_dict = {inputs[0].name: valores}
        # Realizar la predicci
        output = model_ann.run([output.name for output in outputs], feed_dict)
        predictions = output[0][0]
      except Exception as e:
        output = None
        predictions = None
      # Obtener los resultados

      result_lg = models['model_lg'].predict(valores)
      result_dt = models['model_dt'].predict(valores)
      result_knn = models['model_knn'].predict(valores)
      result_svm = models['model_svm'].predict(valores)
      result_nb = models['model_nb'].predict(valores)
      result_gb = models['model_gb'].predict(valores)
      result_rf = models['model_rf'].predict(valores)
      result_ann = [int(predictions[0].round())]
      result_avrg = [
        models['model_lg'].predict(valores),
        models['model_dt'].predict(valores),
        models['model_knn'].predict(valores),
        models['model_svm'].predict(valores),
        models['model_nb'].predict(valores),
        models['model_gb'].predict(valores),
        models['model_rf'].predict(valores), [int(predictions[0].round())]
      ]
      result_avrg = np.array(result_avrg)
      counts = np.bincount(result_avrg.flatten())
      # Encontrar el valor con mayor frecuencia
      result_avrg = np.argmax(counts)
      try:
        if result_lg == [1]:
          result_lg = 'Positive'
        elif result_lg == [0]:
          result_lg = "Negatve"
        if result_dt == [1]:
          result_dt = 'Positive'
        elif result_dt == [0]:
          result_dt = "Negative"
        if result_knn == [1]:
          result_knn = 'Positive'
        elif result_knn == [0]:
          result_knn = "Negative"
        if result_svm == [1]:
          result_svm = 'Positive'
        elif result_svm == [0]:
          result_svm = "Negative"
        if result_nb == [1]:
          result_nb = 'Positive'
        elif result_nb == [0]:
          result_nb = "Negative"
        if result_gb == [1]:
          result_gb = 'Positive'
        elif result_gb == [0]:
          result_gb = "Negative"
        if result_rf == [1]:
          result_rf = 'Positive'
        elif result_rf == [0]:
          result_rf = "Negative"
        if result_ann == [1]:
          result_ann = 'Positive'
        elif result_ann == [0]:
          result_ann = "Negative"
        if result_avrg == [1]:
          result_avrg = 'Positive'
        elif result_avrg == [0]:
          result_avrg = "Negative"
        else:
          result_dt = 'An error ocurred'
          result_lg = 'An error ocurred'
          result_knn = 'An error ocurred'
          result_svm = 'An error ocurred'
          result_nb = 'An error ocurred'
          result_gb = 'An error ocurred'
          result_rf = 'An error ocurred'
          result_ann = 'An error ocurred'
          result_avrg = 'An error ocurred'
          mensaje_personalizado = None
      except ValueError as e:
        mensaje_personalizado = "An error ocurred, please check all the selected information and that there is no missing values in the table data"

      result = {
        'Logistic Regression': result_lg,
        'Decision Tree': result_dt,
        'K-nearest neighbor': result_knn,
        'Support vector machine': result_svm,
        'Naive Bayes': result_nb,
        'Gradient boosted tree': result_gb,
        'Random Forest': result_rf,
        'ANN': result_ann,
        'Combination of all': result_avrg
      }
    elif algoritmo == 'algoritmo9' and enfermedad == 'diabetes':
      valores = np.array(valores).astype(np.float32).reshape(1, -1)
      model_ann = rt.InferenceSession('models/model_ann_db.onnx')
      inputs = model_ann.get_inputs()
      outputs = model_ann.get_outputs()
      feed_dict = {inputs[0].name: valores}
      # Realizar la predicción
      try:
        output = model_ann.run([output.name for output in outputs], feed_dict)
        predictions = output[0][0]
      except Exception as e:
        output = None
        predictions = None
      # Obtener los resultados

      result_lg = models['model_lg_diabetes'].predict(valores)
      result_dt = models['model_dt_diabetes'].predict(valores)
      result_knn = models['model_knn_diabetes'].predict(valores)
      result_svm = models['model_svm_diabetes'].predict(valores)
      result_nb = models['model_nb_diabetes'].predict(valores)
      result_gb = models['model_gb_diabetes'].predict(valores)
      result_rf = models['model_rf_diabetes'].predict(valores)
      result_ann = [int(predictions[0].round())]
      result_avrg = [
        models['model_lg_diabetes'].predict(valores),
        models['model_dt_diabetes'].predict(valores),
        models['model_knn_diabetes'].predict(valores),
        models['model_svm_diabetes'].predict(valores),
        models['model_nb_diabetes'].predict(valores),
        models['model_gb_diabetes'].predict(valores),
        models['model_rf_diabetes'].predict(valores),
        [int(predictions[0].round())]
      ]
      result_avrg = np.array(result_avrg)
      counts = np.bincount(result_avrg.flatten())
      # Encontrar el valor con mayor frecuencia
      result_avrg = np.argmax(counts)
      try:
        if result_lg == [1]:
          result_lg = 'Positive'
        elif result_lg == [0]:
          result_lg = "Negatve"
        if result_dt == [1]:
          result_dt = 'Positive'
        elif result_dt == [0]:
          result_dt = "Negative"
        if result_knn == [1]:
          result_knn = 'Positive'
        elif result_knn == [0]:
          result_knn = "Negative"
        if result_svm == [1]:
          result_svm = 'Positive'
        elif result_svm == [0]:
          result_svm = "Negative"
        if result_nb == [1]:
          result_nb = 'Positive'
        elif result_nb == [0]:
          result_nb = "Negative"
        if result_gb == [1]:
          result_gb = 'Positive'
        elif result_gb == [0]:
          result_gb = "Negative"
        if result_rf == [1]:
          result_rf = 'Positive'
        elif result_rf == [0]:
          result_rf = "Negative"
        if result_ann == [1]:
          result_ann = 'Positive'
        elif result_ann == [0]:
          result_ann = "Negative"
        if result_avrg == [1]:
          result_avrg = 'Positive'
        elif result_avrg == [0]:
          result_avrg = "Negative"
        else:
          result_dt = 'An error ocurred'
          result_lg = 'An error ocurred'
          result_knn = 'An error ocurred'
          result_svm = 'An error ocurred'
          result_nb = 'An error ocurred'
          result_gb = 'An error ocurred'
          result_rf = 'An error ocurred'
          result_ann = 'An error ocurred'
          result_avrg = 'An error ocurred'
          mensaje_personalizado = None
      except ValueError as e:
        mensaje_personalizado = "An error ocurred, please check all the selected information and that there is no missing values in the table data"

      result = {
        'Logistic Regression': result_lg,
        'Decision Tree': result_dt,
        'K-nearest neighbor': result_knn,
        'Support vector machine': result_svm,
        'Naive Bayes': result_nb,
        'Gradient boosted tree': result_gb,
        'Random Forest': result_rf,
        'ANN': result_ann,
        'Combination of all': result_avrg
      }
    elif algoritmo == 'algoritmo9' and enfermedad == 'hepatitis':
      valores = np.array(valores).astype(np.float32).reshape(1, -1)
      model_ann = rt.InferenceSession('models/model_ann_hep.onnx')
      inputs = model_ann.get_inputs()
      outputs = model_ann.get_outputs()
      feed_dict = {inputs[0].name: valores}
      # Realizar la predicción
      try:
        output = model_ann.run([output.name for output in outputs], feed_dict)
        predictions = output[0][0]
      except Exception as e:
        output = None
        predictions = None
      # Obtener los resultados

      result_lg = models['model_lg_hep'].predict(valores)
      result_dt = models['model_dt_hep'].predict(valores)
      result_knn = models['model_knn_hep'].predict(valores)
      result_svm = models['model_svm_hep'].predict(valores)
      result_nb = models['model_nb_hep'].predict(valores)
      result_gb = models['model_gb_hep'].predict(valores)
      result_rf = models['model_rf_hep'].predict(valores)
      result_ann = [int(predictions[0].round())]
      result_avrg = [
        models['model_lg_hep'].predict(valores),
        models['model_dt_hep'].predict(valores),
        models['model_knn_hep'].predict(valores),
        models['model_svm_hep'].predict(valores),
        models['model_nb_hep'].predict(valores),
        models['model_gb_hep'].predict(valores),
        models['model_rf_hep'].predict(valores), [int(predictions[0].round())]
      ]
      result_avrg = np.array(result_avrg)
      counts = np.bincount(result_avrg.flatten())
      # Encontrar el valor con mayor frecuencia
      result_avrg = np.argmax(counts)
      try:
        if result_lg == [1]:
          result_lg = 'Positive'
        elif result_lg == [0]:
          result_lg = "Negatve"
        if result_dt == [1]:
          result_dt = 'Positive'
        elif result_dt == [0]:
          result_dt = "Negative"
        if result_knn == [1]:
          result_knn = 'Positive'
        elif result_knn == [0]:
          result_knn = "Negative"
        if result_svm == [1]:
          result_svm = 'Positive'
        elif result_svm == [0]:
          result_svm = "Negative"
        if result_nb == [1]:
          result_nb = 'Positive'
        elif result_nb == [0]:
          result_nb = "Negative"
        if result_gb == [1]:
          result_gb = 'Positive'
        elif result_gb == [0]:
          result_gb = "Negative"
        if result_rf == [1]:
          result_rf = 'Positive'
        elif result_rf == [0]:
          result_rf = "Negative"
        if result_ann == [1]:
          result_ann = 'Positive'
        elif result_ann == [0]:
          result_ann = "Negative"
        if result_avrg == [1]:
          result_avrg = 'Positive'
        elif result_avrg == [0]:
          result_avrg = "Negative"
        else:
          result_dt = 'An error ocurred'
          result_lg = 'An error ocurred'
          result_knn = 'An error ocurred'
          result_svm = 'An error ocurred'
          result_nb = 'An error ocurred'
          result_gb = 'An error ocurred'
          result_rf = 'An error ocurred'
          result_ann = 'An error ocurred'
          result_avrg = 'An error ocurred'
          mensaje_personalizado = None
      except ValueError as e:
        mensaje_personalizado = "An error ocurred, please check all the selected information and that there is no missing values in the table data"

      result = {
        'Logistic Regression': result_lg,
        'Decision Tree': result_dt,
        'K-nearest neighbor': result_knn,
        'Support vector machine': result_svm,
        'Naive Bayes': result_nb,
        'Gradient boosted tree': result_gb,
        'Random Forest': result_rf,
        'ANN': result_ann,
        'Combination of all': result_avrg
      }
    elif algoritmo == 'algoritmo9' and enfermedad == 'parkinson':
      valores = np.array(valores).astype(np.float32).reshape(1, -1)
      model_ann = rt.InferenceSession('models/model_ann_par.onnx')
      inputs = model_ann.get_inputs()
      outputs = model_ann.get_outputs()
      feed_dict = {inputs[0].name: valores}
      # Realizar la predicción
      try:
        output = model_ann.run([output.name for output in outputs], feed_dict)
        predictions = output[0][0]
      except Exception as e:
        output = None
        predictions = None
      # Obtener los resultados

      result_lg = models['model_lg_par'].predict(valores)
      result_dt = models['model_dt_par'].predict(valores)
      result_knn = models['model_knn_par'].predict(valores)
      result_svm = models['model_svm_par'].predict(valores)
      result_nb = models['model_nb_par'].predict(valores)
      result_gb = models['model_gb_par'].predict(valores)
      result_rf = models['model_rf_par'].predict(valores)
      result_ann = [int(predictions[0].round())]
      result_avrg = [
        models['model_lg_par'].predict(valores),
        models['model_dt_par'].predict(valores),
        models['model_knn_par'].predict(valores),
        models['model_svm_par'].predict(valores),
        models['model_nb_par'].predict(valores),
        models['model_gb_par'].predict(valores),
        models['model_rf_par'].predict(valores), [int(predictions[0].round())]
      ]
      result_avrg = np.array(result_avrg)
      counts = np.bincount(result_avrg.flatten())
      # Encontrar el valor con mayor frecuencia
      result_avrg = np.argmax(counts)
      try:
        if result_lg == [1]:
          result_lg = 'Positive'
        elif result_lg == [0]:
          result_lg = "Negatve"
        if result_dt == [1]:
          result_dt = 'Positive'
        elif result_dt == [0]:
          result_dt = "Negative"
        if result_knn == [1]:
          result_knn = 'Positive'
        elif result_knn == [0]:
          result_knn = "Negative"
        if result_svm == [1]:
          result_svm = 'Positive'
        elif result_svm == [0]:
          result_svm = "Negative"
        if result_nb == [1]:
          result_nb = 'Positive'
        elif result_nb == [0]:
          result_nb = "Negative"
        if result_gb == [1]:
          result_gb = 'Positive'
        elif result_gb == [0]:
          result_gb = "Negative"
        if result_rf == [1]:
          result_rf = 'Positive'
        elif result_rf == [0]:
          result_rf = "Negative"
        if result_ann == [1]:
          result_ann = 'Positive'
        elif result_ann == [0]:
          result_ann = "Negative"
        if result_avrg == [1]:
          result_avrg = 'Positive'
        elif result_avrg == [0]:
          result_avrg = "Negative"
        else:
          result_dt = 'An error ocurred'
          result_lg = 'An error ocurred'
          result_knn = 'An error ocurred'
          result_svm = 'An error ocurred'
          result_nb = 'An error ocurred'
          result_gb = 'An error ocurred'
          result_rf = 'An error ocurred'
          result_ann = 'An error ocurred'
          result_avrg = 'An error ocurred'
          mensaje_personalizado = None
      except ValueError as e:
        mensaje_personalizado = "An error ocurred, please check all the selected information and that there is no missing values in the table data"

      result = {
        'Logistic Regression': result_lg,
        'Decision Tree': result_dt,
        'K-nearest neighbor': result_knn,
        'Support vector machine': result_svm,
        'Naive Bayes': result_nb,
        'Gradient boosted tree': result_gb,
        'Random Forest': result_rf,
        'ANN': result_ann,
        'Combination of all': result_avrg
      }
    elif algoritmo == 'algoritmo9' and enfermedad == 'HCC':
      valores = np.array(valores).astype(np.float32).reshape(1, -1)
      model_ann = rt.InferenceSession('models/model_ann_hcc.onnx')
      inputs = model_ann.get_inputs()
      outputs = model_ann.get_outputs()
      feed_dict = {inputs[0].name: valores}
      # Realizar la predicción
      try:
        output = model_ann.run([output.name for output in outputs], feed_dict)
        predictions = output[0][0]
      except Exception as e:
        output = None
        predictions = None
      # Obtener los resultados

      result_lg = models['model_lg_hcc'].predict(valores)
      result_dt = models['model_dt_hcc'].predict(valores)
      result_knn = models['model_knn_hcc'].predict(valores)
      result_svm = models['model_svm_hcc'].predict(valores)
      result_nb = models['model_nb_hcc'].predict(valores)
      result_gb = models['model_gb_hcc'].predict(valores)
      result_rf = models['model_rf_hcc'].predict(valores)
      result_ann = [int(predictions[0].round())]
      result_avrg = [
        models['model_lg_hcc'].predict(valores),
        models['model_dt_hcc'].predict(valores),
        models['model_knn_hcc'].predict(valores),
        models['model_svm_hcc'].predict(valores),
        models['model_nb_hcc'].predict(valores),
        models['model_gb_hcc'].predict(valores),
        models['model_rf_hcc'].predict(valores), [int(predictions[0].round())]
      ]
      result_avrg = np.array(result_avrg)
      counts = np.bincount(result_avrg.flatten())
      # Encontrar el valor con mayor frecuencia
      result_avrg = np.argmax(counts)
      try:
        if result_lg == [1]:
          result_lg = 'Positive'
        elif result_lg == [0]:
          result_lg = "Negatve"
        if result_dt == [1]:
          result_dt = 'Positive'
        elif result_dt == [0]:
          result_dt = "Negative"
        if result_knn == [1]:
          result_knn = 'Positive'
        elif result_knn == [0]:
          result_knn = "Negative"
        if result_svm == [1]:
          result_svm = 'Positive'
        elif result_svm == [0]:
          result_svm = "Negative"
        if result_nb == [1]:
          result_nb = 'Positive'
        elif result_nb == [0]:
          result_nb = "Negative"
        if result_gb == [1]:
          result_gb = 'Positive'
        elif result_gb == [0]:
          result_gb = "Negative"
        if result_rf == [1]:
          result_rf = 'Positive'
        elif result_rf == [0]:
          result_rf = "Negative"
        if result_ann == [1]:
          result_ann = 'Positive'
        elif result_ann == [0]:
          result_ann = "Negative"
        if result_avrg == [1]:
          result_avrg = 'Positive'
        elif result_avrg == [0]:
          result_avrg = "Negative"
        else:
          result_dt = 'An error ocurred'
          result_lg = 'An error ocurred'
          result_knn = 'An error ocurred'
          result_svm = 'An error ocurred'
          result_nb = 'An error ocurred'
          result_gb = 'An error ocurred'
          result_rf = 'An error ocurred'
          result_ann = 'An error ocurred'
          result_avrg = 'An error ocurred'
          mensaje_personalizado = None
      except ValueError as e:
        mensaje_personalizado = "An error ocurred, please check all the selected information and that there is no missing values in the table data"

      result = {
        'Logistic Regression': result_lg,
        'Decision Tree': result_dt,
        'K-nearest neighbor': result_knn,
        'Support vector machine': result_svm,
        'Naive Bayes': result_nb,
        'Gradient boosted tree': result_gb,
        'Random Forest': result_rf,
        'ANN': result_ann,
        'Combination of all': result_avrg
      }
    else:
      if modelo_seleccionado is not None and algoritmo != 'algoritmo8':
        result = modelo_seleccionado.predict([valores])
  except ValueError as e:
    result = 'An error ocurred, please check all the selected information and that there is no missing values in the table data'
  try:
    if result == [1]:
      result = 'positive diagnostic, please consider consulting a doctor'
    elif result == [0]:
      result = " negative diagnostic, all good!"
    elif algoritmo == 'algoritmo9':
      result = result
    else:
      result = 'An error ocurred, please check all the selected information and that there is no missing values in the table data'
    mensaje_personalizado = None
  except ValueError as e:
    result = 'An error ocurred, please check all the selected information and that there is no missing values in the table data'
  if algoritmo == 'algoritmo1':
    algorithm = 'Logistic Regression'
  elif algoritmo == 'algoritmo2':
    algorithm = 'Decision Tree'
  elif algoritmo == 'algoritmo3':
    algorithm = 'K-nearest neighbor'
  elif algoritmo == 'algoritmo4':
    algorithm = 'Support vector machine'
  elif algoritmo == 'algoritmo5':
    algorithm = 'Naive Bayes'
  elif algoritmo == 'algoritmo6':
    algorithm = 'Gradient boosted tree'
  elif algoritmo == 'algoritmo7':
    algorithm = 'Random forest'
  elif algoritmo == 'algoritmo8':
    algorithm = 'Artificial Neural Network'
  elif algoritmo == 'algoritmo9':
    algorithm = 'All algorithms'

  return render_template('result.html',
                         results=result,
                         error=mensaje_personalizado,
                         algorithm=algorithm,
                         data=valores)


if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)
