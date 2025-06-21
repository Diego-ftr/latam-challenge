import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Union, List, Optional
import pickle
import os
from pathlib import Path
import logging

# Configurar logging básico si no está configurado por la aplicación madre
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DelayModel:

    def __init__(
        self,
        auto_train_for_tests: bool = True  # Parámetro para compatibilidad con tests
    ):
        self._model = None  # El modelo se debe guardar en este atributo
        self._auto_train_for_tests = auto_train_for_tests
        # Usar ruta absoluta para evitar problemas cuando se ejecuta desde diferentes directorios
        self._base_dir = Path(__file__).resolve().parent
        self._model_path = self._base_dir / "model.pkl"
        self._threshold_in_minutes = 15  # Umbral para considerar un vuelo como retrasado
        self._top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        # Intentar cargar modelo existente al inicializar si está disponible
        self.load_model()
        

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepara los datos crudos para entrenamiento o predicción.

        Args:
            data (pd.DataFrame): datos crudos.
            target_column (str, optional): si se especifica, retorna el target.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features y target.
            o
            pd.DataFrame: solo features.
        """
        # Crear una copia para evitar modificar los datos originales
        data = data.copy()
        
        # Funciones de ingeniería de features del notebook
        def get_period_day(date):
            date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
            morning_min = datetime.strptime("05:00", '%H:%M').time()
            morning_max = datetime.strptime("11:59", '%H:%M').time()
            afternoon_min = datetime.strptime("12:00", '%H:%M').time()
            afternoon_max = datetime.strptime("18:59", '%H:%M').time()
            evening_min = datetime.strptime("19:00", '%H:%M').time()
            evening_max = datetime.strptime("23:59", '%H:%M').time()
            night_min = datetime.strptime("00:00", '%H:%M').time()
            night_max = datetime.strptime("04:59", '%H:%M').time()
            
            if(date_time >= morning_min and date_time <= morning_max):
                return 'mañana'
            elif(date_time >= afternoon_min and date_time <= afternoon_max):
                return 'tarde'
            elif(
                (date_time >= evening_min and date_time <= evening_max) or
                (date_time >= night_min and date_time <= night_max)
            ):
                return 'noche'
        
        def is_high_season(fecha):
            fecha_año = int(fecha.split('-')[0])
            fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
            range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
            range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
            range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
            range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
            range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
            range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
            range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
            range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
            
            if ((fecha >= range1_min and fecha <= range1_max) or 
                (fecha >= range2_min and fecha <= range2_max) or 
                (fecha >= range3_min and fecha <= range3_max) or
                (fecha >= range4_min and fecha <= range4_max)):
                return 1
            else:
                return 0
        
        def get_min_diff(data):
            fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
            fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
            min_diff = ((fecha_o - fecha_i).total_seconds())/60
            return min_diff
        
        # Aplicar ingeniería de features
        data['period_day'] = data['Fecha-I'].apply(get_period_day)
        data['high_season'] = data['Fecha-I'].apply(is_high_season)
        data['min_diff'] = data.apply(get_min_diff, axis = 1)
        
        # Crear columna delay si se especifica target_column
        if target_column:
            data['delay'] = np.where(data['min_diff'] > self._threshold_in_minutes, 1, 0)
        
        # Crear variables dummy
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix='MES')], 
            axis = 1
        )
        
        # Solo mantener las top 10 features, pero asegurar que todas estén presentes
        features_to_add = []
        for feature in self._top_10_features:
            if feature not in features.columns:
                features_to_add.append(feature)
        
        # Agregar features faltantes como columnas de ceros
        if features_to_add:
            for feature in features_to_add:
                features[feature] = 0
                
        # Seleccionar solo top 10 features usando reindex para asegurar orden
        features = features.reindex(self._top_10_features, axis=1, fill_value=0)
        
        if target_column:
            target = data[['delay']]
            return features, target
        else:
            return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Entrena el modelo con datos preprocesados.

        Args:
            features (pd.DataFrame): datos preprocesados.
            target (pd.DataFrame): variable objetivo.
        """
        # Convertir target a array para el entrenamiento
        target_array = target.values.ravel()
        
        # Crear y entrenar el modelo con pesos de clases balanceados
        self._model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            random_state=1
        )
        self._model.fit(features, target_array)
        
        # Guardar el modelo después del entrenamiento
        self.save_model()

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predice retrasos para nuevos vuelos.

        Args:
            features (pd.DataFrame): datos preprocesados.
        
        Returns:
            (List[int]): predicciones.
        """
        if self._model is None:
            # Solo para tests: entrenar automáticamente si está habilitado
            if self._auto_train_for_tests:
                try:
                    csv_path = self._base_dir.parent / "data" / "data.csv"
                    if csv_path.exists():
                        logging.info(f"Entrenando modelo automáticamente desde {csv_path}")
                        data = pd.read_csv(csv_path, dtype={'Vlo-I': 'object', 'Vlo-O': 'object'})
                        features_train, target_train = self.preprocess(data, target_column="delay")
                        self.fit(features_train, target_train)
                    else:
                        raise ValueError(f"No se encontró el archivo de datos en {csv_path}")
                except Exception as e:
                    logging.error(f"Error al entrenar modelo automáticamente: {str(e)}")
                    raise ValueError(f"El modelo no ha sido entrenado. Error: {str(e)}")
            else:
                raise ValueError("El modelo no ha sido entrenado. Llame a fit() antes de predict() o asegúrese de que existe un modelo guardado.")
        predictions = self._model.predict(features)
        return predictions.tolist()
    
    def save_model(self) -> None:
        """Guarda el modelo entrenado en disco."""
        if self._model is not None:
            # Asegurar que el directorio existe
            self._model_path.parent.mkdir(exist_ok=True)
            with open(self._model_path, 'wb') as file:
                pickle.dump(self._model, file)
    
    def load_model(self) -> None:
        """Carga un modelo entrenado desde disco."""
        try:
            if self._model_path.exists():
                with open(self._model_path, 'rb') as file:
                    self._model = pickle.load(file)
        except (pickle.UnpicklingError, FileNotFoundError, EOFError) as e:
            # Si la carga falla, el modelo permanece None
            logging.warning(f"No se pudo cargar el modelo desde {self._model_path}: {str(e)}")
            pass