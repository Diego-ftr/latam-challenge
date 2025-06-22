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

### Service Account Configuration
- **Email**: latam-challenge-diegoftr@plenary-justice-357523.iam.gserviceaccount.com
- **Permisos configurados**:
  - Administrador de Cloud Run
  - Editor de Cloud Build
  - Lector de Artifact Registry
  - Visualizador de objetos de Storage
  - Service Account User

### GitHub Secrets Configurados
- `GCP_SERVICE_ACCOUNT_KEY`: Clave JSON de la service account
- `GCP_PROJECT_ID`: plenary-justice-357523
- `GCP_REGION`: us-central1

## 3.2 Containerización con Docker

### Multi-Stage Dockerfile Optimizado

Se implementó un Dockerfile multi-stage para optimizar el tamaño de la imagen y el tiempo de despliegue:

**Stage 1 (Builder)**:
- Instalación de dependencias de compilación
- Entrenamiento del modelo en build time
- Generación de model.pkl

**Stage 2 (Runtime)**:
- Imagen base mínima
- Solo dependencias de runtime
- Copia del modelo pre-entrenado
- Configuración para Cloud Run (puerto 8080)

**Beneficios**:
- Imagen final más pequeña (sin data.csv ni toolchain)
- Arranque más rápido (modelo pre-entrenado)
- Mejor performance en Cold Start de Cloud Run

## 3.3 Configuración de Cloud Run

### Especificaciones del Servicio
- **Nombre**: latam-challenge
- **Puerto**: 8080
- **Memoria**: 1Gi
- **CPU**: 1
- **Max instances**: 10
- **Min instances**: 0 (auto-scaling completo)
- **Concurrency**: 80
- **Acceso**: Público (allow-unauthenticated)

### Artifact Registry
Se configuró Artifact Registry en lugar de Container Registry (deprecado):
- Repository: `latam-challenge-repo`
- Formato: Docker
- Ubicación: Regional según GCP_REGION

## 3.4 Deployment Status

**Estado actual**: Configuración completa, pendiente de deployment
- ✅ Dockerfile multi-stage optimizado
- ✅ Service Account configurada
- ✅ Secrets de GitHub establecidos
- ✅ Workflows de CI/CD preparados
- 🔄 Pendiente: Push a main para activar deployment automático

---

# PARTE IV: CI/CD Implementation

## 4.1 Pipeline de Integración Continua (ci.yml)

### Configuración del Workflow CI
- **Triggers**: Push a main/develop/feature/* y Pull Requests
- **Runner**: ubuntu-latest
- **Python**: 3.10

### Pipeline Steps
1. **Checkout**: Obtener código fuente
2. **Setup Python**: Configurar entorno Python 3.10
3. **Install Dependencies**: Solo requirements.txt + requirements-test.txt (optimizado)
4. **Run Model Tests**: `make model-test`
5. **Run API Tests**: `make api-test`
6. **Upload Coverage**: Subir reportes a Codecov

### Optimizaciones Implementadas
- Removido `requirements-dev.txt` para acelerar builds
- Solo dependencias esenciales para testing
- Cache de dependencias implícito en GitHub Actions

## 4.2 Pipeline de Entrega Continua (cd.yml)

### Configuración del Workflow CD
- **Trigger**: Push a main branch únicamente
- **Runner**: ubuntu-latest
- **Target**: Google Cloud Run

### Pipeline Steps
1. **Checkout**: Obtener código fuente
2. **Setup Cloud SDK**: Autenticación con GCP usando service account
3. **Configure Artifact Registry**: Setup para docker push
4. **Build & Push Image**: 
   - Build multi-stage Docker image
   - Push a Artifact Registry
   - Tag con SHA del commit
5. **Deploy to Cloud Run**:
   - Deploy automático con configuración optimizada
   - Rolling updates sin downtime

### Mejoras Técnicas
- **Artifact Registry** en lugar de Container Registry deprecado
- **Multi-stage builds** para imágenes optimizadas
- **Configuración automática** de Docker auth para Artifact Registry
- **Tagging con SHA** para tracking de versiones

## 4.3 Workflow Security & Best Practices

### Security Measures
- Service Account con permisos mínimos necesarios
- Secrets almacenados en GitHub Secrets (no hardcoded)
- Autenticación con JSON key temporal

### Best Practices Implementadas
- Workflows separados para CI y CD
- CD solo en main branch (production)
- Tests obligatorios antes de deployment
- Rollback automático en caso de fallo
- Logging completo para debugging

