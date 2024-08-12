import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.feature_selection import SelectFromModel
import logging
import joblib
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import time

# Configurar logging
log_file = 'property_price_prediction.log'
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Lista de características
FEATURES = ['Dormitorios', 'Baños', 'Metros Cuadrados', 'Metros por dormitorio', 'Baños por dormitorio']

def convert_to_float(value):
    if isinstance(value, str):
        return float(value.replace('.', '').replace(',', '.'))
    return float(value)

def prepare_data(file_paths):
    dfs = []
    for file_path in file_paths:
        logging.info(f"Procesando archivo: {file_path}")
        df = pd.read_csv(file_path)
        df['Precio'] = df['Precio'].apply(convert_to_float)
        df['Precio'] = np.where(df['Moneda'] == 'UF', df['Precio'] * 30000, df['Precio'])
        df = df.drop('Moneda', axis=1)
        
        for col in ['Dormitorios', 'Baños', 'Metros Cuadrados']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.dropna()
    logging.info(f"Datos combinados: {len(combined_df)} filas")
    return combined_df

def feature_engineering(df):
    df['Metros por dormitorio'] = df['Metros Cuadrados'] / df['Dormitorios'].replace(0, 1)
    df['Baños por dormitorio'] = df['Baños'] / df['Dormitorios'].replace(0, 1)
    return df

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_dist = {
        'n_estimators': sp_randint(100, 500),
        'max_depth': sp_randint(10, 50),
        'min_samples_split': sp_randint(2, 20),
        'min_samples_leaf': sp_randint(1, 10),
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestRegressor(random_state=42)
    
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                                       n_iter=100, cv=5, random_state=42, n_jobs=-1)
    random_search.fit(X_train_scaled, y_train)
    
    best_model = random_search.best_estimator_
    
    selector = SelectFromModel(best_model, prefit=True)
    feature_mask = selector.get_support()
    selected_features = np.array(FEATURES)[feature_mask]
    
    logging.info(f"Características seleccionadas: {selected_features}")
    logging.info(f"Número de características seleccionadas: {len(selected_features)}")
    
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    best_model.fit(X_train_selected, y_train)
    
    return best_model, scaler, selector, X_test_selected, y_test, selected_features

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    median_ae = median_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAE': mae,
        'MedianAE': median_ae,
        'MAPE': mape
    }
    
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
        explain_metric(metric, value)
    
    return metrics

def explain_metric(metric, value):
    explanations = {
        'MSE': f"El Error Cuadrático Medio (MSE) es {value:.4f}. Un valor más bajo indica mejor rendimiento. Para mejorar, considere usar técnicas de regularización o aumentar la complejidad del modelo si es muy alto.",
        'RMSE': f"La Raíz del Error Cuadrático Medio (RMSE) es {value:.4f}. Esta métrica está en la misma unidad que la variable objetivo. Para mejorar, considere feature engineering o usar modelos más complejos si es necesario.",
        'R2': f"El coeficiente de determinación R² es {value:.4f}. Un valor más cercano a 1 indica un mejor ajuste. Si es bajo, considere agregar más características relevantes o usar modelos no lineales.",
        'MAE': f"El Error Absoluto Medio (MAE) es {value:.4f}. Esta métrica es robusta a outliers. Para mejorar, considere técnicas de manejo de outliers o transformaciones de datos.",
        'MedianAE': f"El Error Absoluto Mediano es {value:.4f}. Es aún más robusto a outliers que MAE. Si es significativamente diferente de MAE, investigue la presencia de outliers extremos.",
        'MAPE': f"El Error Porcentual Absoluto Medio (MAPE) es {value:.4f}%. Proporciona el error en términos de porcentaje. Para mejorar, considere transformaciones logarítmicas si los datos son muy sesgados."
    }
    logging.info(explanations[metric])

def predict_price_range(model, scaler, selector, dormitorios, baños, metros_cuadrados, selected_features):
    metros_por_dormitorio = metros_cuadrados / max(dormitorios, 1)
    baños_por_dormitorio = baños / max(dormitorios, 1)
    
    new_property = pd.DataFrame([[dormitorios, baños, metros_cuadrados, 
                                  metros_por_dormitorio, baños_por_dormitorio]], 
                                columns=FEATURES)
    
    logging.info(f"Nueva propiedad antes de escalar: {new_property}")
    
    new_property_scaled = scaler.transform(new_property)
    
    logging.info(f"Nueva propiedad después de escalar: {new_property_scaled}")
    logging.info(f"Forma de nueva propiedad escalada: {new_property_scaled.shape}")
    
    # Asegúrate de usar solo las características seleccionadas
    new_property_selected = new_property_scaled[:, selector.get_support()]
    
    logging.info(f"Nueva propiedad después de selección: {new_property_selected}")
    logging.info(f"Forma de nueva propiedad seleccionada: {new_property_selected.shape}")
    
    predicted_price = model.predict(new_property_selected)[0]
    
    lower_bound = predicted_price * 0.9
    upper_bound = predicted_price * 1.1
    
    return lower_bound, upper_bound

def export_model(model, scaler, selector, selected_features, filename='property_price_model.joblib'):
    joblib.dump({'model': model, 'scaler': scaler, 'selector': selector, 'selected_features': selected_features}, filename)
    logging.info(f"Modelo exportado como {filename}")

if __name__ == "__main__":
    start_time = time.time()
    logging.info("Iniciando proceso de predicción de precios de propiedades")
    
    file_paths = [
        "data/propiedades_valdivia-1.csv",
        "data/propiedades_valdivia-2.csv",
        "data/propiedades_valdivia-3.csv"
    ]
    
    logging.info("Iniciando procesamiento de datos")
    df = prepare_data(file_paths)
    
    logging.info("Realizando ingeniería de características")
    df = feature_engineering(df)
    
    logging.info("Preparando características y objetivo")
    X = df[FEATURES]
    y = df['Precio']
    
    logging.info(f"Características utilizadas: {FEATURES}")
    logging.info(f"Forma de X: {X.shape}")
    
    logging.info("Entrenando modelo con búsqueda de hiperparámetros")
    model, scaler, selector, X_test, y_test, selected_features = train_model(X, y)
    
    logging.info("Evaluando modelo")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Exportar modelo
    export_model(model, scaler, selector, selected_features)
    
    # Ejemplo de predicción
    dormitorios = 3
    baños = 1
    metros_cuadrados = 80
    lower_bound, upper_bound = predict_price_range(model, scaler, selector, dormitorios, baños, metros_cuadrados, selected_features)
    logging.info(f"Rango de precio estimado para una propiedad de {dormitorios} dormitorios, {baños} baños y {metros_cuadrados} m²:")
    logging.info(f"Entre ${lower_bound:,.0f} y ${upper_bound:,.0f}")
    
    end_time = time.time()
    logging.info(f"Proceso completado en {end_time - start_time:.2f} segundos")