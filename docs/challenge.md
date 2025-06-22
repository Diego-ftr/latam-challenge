# LATAM Challenge - Software Engineer (ML & LLMs)

## Resumen Ejecutivo

Este documento presenta la implementación completa del challenge LATAM para el rol de Software Engineer especializado en ML y LLMs. El proyecto involucra la operacionalización de un modelo de predicción de retrasos de vuelos desarrollado por el equipo de Data Science, incluyendo transcripción del modelo, implementación de API, despliegue en cloud y configuración de CI/CD.

---

# PARTE I: Operacionalización del Modelo

## 1.1 Análisis del Notebook de Exploración

### Configuración del Entorno de Desarrollo

Para mantener las dependencias organizadas y evitar conflictos, se configuró un entorno virtual de Python 3.10:

```bash
python3.10 -m venv .venv
source .venv/Scripts/activate  # En Windows
pip install -r requirements-dev.txt
pip install -r requirements-test.txt
pip install -r requirements.txt
```

### Bugs Identificados y Corregidos

#### 1.1.1 Error en la Función `is_high_season`
- **Problema**: La función no considera correctamente los límites de tiempo. Por ejemplo, `is_high_season("2017-12-31 14:55:00")` devolvía 0 cuando debería devolver 1.
- **Causa**: Los rangos de fecha terminaban en "YYYY-MM-DD 00:00:00", excluyendo vuelos posteriores en el mismo día final.
- **Solución**: Ajustar la función para usar comparaciones inclusivas (`<=`) en lugar de exclusivas (`<`) para el límite superior.

#### 1.1.2 Dependencia Faltante: XGBoost ✅ RESUELTO
- **Problema**: El notebook intenta importar `xgboost` pero no está instalado.
- **Error**: `ModuleNotFoundError: No module named 'xgboost'`
- **Solución**: Agregué `xgboost~=1.7.0` a requirements-dev.txt
- **Resultado**: XGBoost funciona correctamente y se pueden ejecutar todos los modelos

#### 1.1.3 Error en Argumentos de `sns.barplot()`
- **Problema**: Todas las llamadas a `sns.barplot()` usan argumentos posicionales que causan TypeError.
- **Solución**: Usar argumentos explícitos `x=` e `y=` en lugar de posicionales

#### 1.1.4 Error en la Función `get_rate_from_column`
- **Problema**: La función calcula incorrectamente la tasa de retraso. Divide el total por los retrasos en lugar de los retrasos por el total.
- **Impacto**: Las tasas de retraso mostradas están invertidas y no representan porcentajes correctos.

#### 1.1.5 Error en la Función `get_period_day`
- **Problema**: La función no retorna valor para algunos casos edge.
- **Solución**: Agregar límites inclusivos y manejo completo de casos.

## 1.2 Análisis y Selección del Modelo

### Visión General

Se realizó un análisis exhaustivo de varios modelos candidatos para predecir retrasos de vuelos. El desafío principal es el desbalance de clases: ~81% vuelos a tiempo vs ~19% retrasados.

Modelos evaluados:
- XGBoost
- Regresión Logística

Configuraciones evaluadas:
- Sin balance de clases
- Con selección de features importantes
- Con/Sin balance de clases combinado con selección de features

### Métricas Consideradas

Se enfocó principalmente en precision, recall y F1-score para la clase retrasada (clase 1):
- **Precision**: Qué tan seguido el modelo acierta cuando predice un retraso
- **Recall**: Cuántos de los retrasos reales identifica el modelo
- **F1-score**: Combina precision y recall, métrica balanceada para problemas desbalanceados

### Resultados Empíricos de los Modelos

#### Modelos Sin Balance de Clases (Todas las Features)
- **XGBoost sin balance**: F1-score clase 1: **0.04**, Recall clase 1: 0.02
- **Regresión Logística sin balance**: F1-score clase 1: **0.06**, Recall clase 1: 0.03

#### Modelos con Top 10 Features y Balance de Clases
- **XGBoost top 10 con balance**: F1-score clase 1: **0.37**, Recall clase 1: 0.69
- **Regresión Logística top 10 con balance**: F1-score clase 1: **0.36**, Recall clase 1: 0.69

### Decisión Final del Modelo

**Selección: Regresión Logística con balance de clases y top 10 features**

**Justificación:**
1. **Rendimiento equivalente**: F1-score de 0.36 vs 0.37 (diferencia insignificante)
2. **Mayor interpretabilidad**: Crucial para stakeholders del negocio
3. **Menor overhead operacional**: Más fácil de mantener y actualizar
4. **Menor riesgo**: Menos dependencias externas y mayor estabilidad
5. **Cumple criterios del challenge**: Recall clase 1 = 0.69 > 0.60, F1-score clase 1 = 0.36 > 0.30

### Features Más Importantes
Las 10 features más relevantes identificadas:
1. OPERA_Latin American Wings
2. MES_7 (Julio)
3. MES_10 (Octubre)
4. OPERA_Grupo LATAM
5. MES_12 (Diciembre)
6. TIPOVUELO_I (Internacional)
7. MES_4 (Abril)
8. MES_11 (Noviembre)
9. OPERA_Sky Airline
10. OPERA_Copa Air

## 1.3 Implementación en model.py

### Mejoras Técnicas Implementadas
- Métodos `save_model()` y `load_model()` para persistencia del modelo
- Guardado automático después del entrenamiento
- Rutas absolutas con pathlib para evitar problemas de directorios
- LogisticRegression con `class_weight='balanced'`
- Parámetro `auto_train_for_tests` para compatibilidad con tests vs producción
- Umbral de retraso (15 minutos) como variable de clase configurable
- Manejo de excepciones con logging específico
- Documentación en español para consistencia

### Configuración del Modelo
- `max_iter=1000` para evitar warnings de convergencia
- `solver='lbfgs'` para optimización eficiente
- `class_weight='balanced'` para manejar el desbalance de clases
- `random_state=1` para reproducibilidad

## 1.4 Resultados de Tests - Parte I ✅

**Estado**: 4 passed, 4 warnings
- **Cobertura**: 82% en model.py (110 statements, 20 missed)
- **Tiempo**: ~37 segundos (incluye entrenamiento del modelo)

---

# PARTE II: Implementación de API con FastAPI

## 2.1 Desarrollo del Endpoint /predict

Se implementó una API REST usando FastAPI que expone el modelo de predicción de retrasos como servicio web.

### Estructura del Endpoint

**URL**: `POST /predict`

**Input**:
```json
{
  "flights": [
    {
      "OPERA": "Aerolineas Argentinas",
      "TIPOVUELO": "N",
      "MES": 3
    }
  ]
}
```

**Output**:
```json
{
  "predict": [0]
}
```

## 2.2 Validación de Datos con Pydantic

Se implementó validación automática de datos de entrada:

1. **MES**: Debe estar entre 1 y 12
2. **TIPOVUELO**: Debe ser 'N' (Nacional) o 'I' (Internacional)
3. **OPERA**: Debe ser una aerolínea válida de la lista predefinida

## 2.3 Problemas Identificados y Solucionados

### Incompatibilidad de Versiones - anyio ✅ RESUELTO
- **Problema**: Error `AttributeError: module 'anyio' has no attribute 'start_blocking_portal'`
- **Causa**: Conflicto de versiones entre FastAPI/Starlette y anyio>=4.0
- **Solución**: Fijar `anyio>=3.7,<4` en requirements-test.txt

### Códigos de Error HTTP ✅ RESUELTO
- **Problema**: Pydantic devuelve 422 por defecto para errores de validación, pero los tests esperan 400
- **Solución**: Implementar custom exception handler para cambiar 422 a 400

## 2.4 Arquitectura de la API

1. **Inicialización del Modelo**: El modelo se carga una vez al arrancar la aplicación
2. **Preprocesamiento**: Se agregan fechas dummy para compatibilidad con el modelo
3. **Predicción**: Se usa el modelo entrenado para generar predicciones
4. **Manejo de Errores**: 
   - 400: Errores de validación de datos
   - 500: Errores internos del servidor
5. **Logging**: Registro de eventos para debugging y monitoreo

## 2.5 Resultados de Tests - Parte II ✅

**Estado**: 4 passed en 6.21s
- **Coverage Total**: 77% (172 statements, 40 missed)
- **Coverage API**: 88% (60 statements, 7 missed)
- **Coverage Model**: 70% (110 statements, 33 missed)

Tests verificados:
- ✅ Predicción exitosa con datos válidos
- ✅ Validación de MES fuera de rango >12
- ✅ Validación de TIPOVUELO inválido
- ✅ Validación de OPERA no reconocida

---

# PARTE III: Despliegue en Cloud (Google Cloud Platform)

## 3.1 Configuración de Google Cloud Platform

### Service Account
- **Email**: latam-challenge-diegoftr@plenary-justice-357523.iam.gserviceaccount.com  
- **Permisos**  
  - Cloud Run Admin  
  - Cloud Build Editor  
  - Artifact Registry Reader  
  - Storage Object Viewer  
  - Service Account User  

### GitHub Secrets
| Secret | Valor |
|--------|-------|
| `GCP_SERVICE_ACCOUNT_KEY` | JSON de la service account |
| `GCP_PROJECT_ID` | `plenary-justice-357523` |
| `GCP_REGION` | `us-central1` |

---

## 3.2 Containerización con Docker (multi-stage)

| Stage | Contenido | Propósito |
|-------|-----------|-----------|
| **Builder** | Dependencias de compilación + `data/` | Entrena el modelo y genera `model.pkl`. |
| **Runtime** | Imagen slim + solo libs de producción | Copia `model.pkl` y expone la API en el puerto 8080. |

**Ventajas**  
- Imagen final ligera (≈ 240 MB).  
- Cold-start más rápido en Cloud Run.

---

## 3.3 Cloud Run

| Parámetro | Valor |
|-----------|-------|
| **Servicio** | `latam-challenge` |
| Región | `us-central1` |
| URL Pública | **https://latam-challenge-630883832403.us-central1.run.app** |
| CPU / Mem | 1 vCPU · 1 GiB |
| Autoscaling | 0 – 10 instancias |
| Concurrency | 80 |
| Auth | `--allow-unauthenticated` |

Las imágenes se almacenan en **Artifact Registry**  
`us-central1-docker.pkg.dev/plenary-justice-357523/latam-challenge-repo/delay-api:<sha>`.

---

## 3.4 Estado de Deployment

- ✅ Dockerfile multi-stage listo  
- ✅ Workflows CI/CD en producción  
- ✅ Push a `main` despliega automáticamente (Cloud Run revision `latam-challenge-00001-85j`)  
- ✅ Servicio operativo en la URL indicada

### Stress-test

`make stress-test` · Locust 1.6 · 100 users · 60 s

| Métrica | Valor |
|---------|-------|
| Peticiones | **5 287** |
| Errores | **0** |
| Throughput medio | **88.5 req/s** |
| Latencia media | **338 ms** |
| p95 | **670 ms** |
| p99 | **750 ms** |

> El servicio sostuvo ~90 req/s sin errores. Con 2 vCPU o `min-instances=1` podría bajarse el p95, pero dentro de los objetivos del reto el rendimiento es adecuado.

---

# PARTE IV: CI/CD

## 4.1 Integración Continua (`ci.yml`)

- **Triggers**: pushes a `main`, `develop`, `feature/*` y PRs.  
- **Pasos clave**  
  1. Checkout  
  2. Python 3.10  
  3. Instalación de `requirements.txt` + `requirements-test.txt`  
     - *Hotfix*: `itsdangerous==1.1.0` fijado para compatibilidad con Locust  
  4. `make model-test` y `make api-test`  
  5. Subida de cobertura a Codecov  

## 4.2 Entrega Continua (`cd.yml`)

1. **Auth** con `google-github-actions/auth@v1`.  
2. `setup-gcloud@v1` (proyecto y CLI).  
3. `gcloud auth configure-docker us-central1-docker.pkg.dev`.  
4. Build & push de la imagen multi-stage etiquetada con `github.sha`.  
5. `gcloud run deploy latam-challenge …` (rolling update sin downtime).

## 4.3 Buenas prácticas

- Service Account con principio de mínimo privilegio.  
- Secrets en GitHub Secrets, nunca hard-coded.  
- Workflows separados (CI - pruebas / CD - despliegue).  
- CD sólo en `main` → evita despliegues accidentales.  
- Rollback automático si el deploy falla.

---
