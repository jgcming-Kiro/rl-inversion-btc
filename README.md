# Proyecto de inversión con aprendizaje por refuerzo

Código y resultados de Q-learning tabular y DQN frente a comprar y mantener, aleatorio y efectivo. Incluye dos recompensas, tres semillas y una división temporal de entrenamiento, validación y prueba. El informe ejecutado está en [results/main/INFORME.md](results/main/INFORME.md).

El [informe HTML](results/main/INFORME.html) incluye los gráficos y puede descargarse y abrirse sin conexión. En la prueba de 2024, el DQN seleccionado en validación obtuvo un retorno medio de 89,49 %, frente a 111,76 % de comprar y mantener. Estos resultados corresponden a esta ventana y a los supuestos del simulador.

## Instalación

Versión ejecutada: Python 3.9.6 en macOS, CPU. Las versiones de las bibliotecas están fijadas en requirements.txt y en el manifiesto de cada ejecución. En otros equipos pueden existir pequeñas diferencias numéricas; se fijan semillas y se activa el cálculo determinista de PyTorch.

```bash
cd Proyecto_RL_Inversion
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows, activa con `.venv\Scripts\activate`. Si tu sistema no ofrece una distribución de PyTorch compatible con Python 3.9, usa Python 3.10 u 3.11 y conserva las versiones de las bibliotecas. No es necesaria GPU.

## Reproducir

```bash
python -m unittest discover -s tests -v
python -m src.train --config configs/default.json --output results/replica_01
```

El segundo comando entrena 12 modelos: dos algoritmos, dos recompensas y tres semillas. Evalúa validación, escribe selection_frozen.json y después recorre prueba. Ejecuta también todos los baselines, guarda curvas y redacta el informe con las métricas calculadas. Usa una carpeta de salida nueva por ejecución; no sobrescribe los resultados entregados. El tiempo depende de la CPU. La reutilización de datos en Q-learning es mayor que en DQN; no se afirma igualdad de presupuesto.

Si usas Jupyter, abre notebooks/eda.ipynb para el análisis y notebooks/proyecto_inversion.ipynb para el recorrido completo. Jupyter es una interfaz opcional; los comandos anteriores ejecutan todo el experimento sin ella. En una instalación nueva puedes añadirla con `python -m pip install notebook`.

## Volver a evaluar modelos guardados, incluidos los baselines

```bash
python -m src.reevaluate --run results/main --output results/auditoria_01
```

Este comando no entrena ni selecciona candidatos. Verifica el hash del CSV y la configuración conservada, carga el escalador y los modelos y repite los recorridos finales y los baselines. Guarda nuevas métricas para contrastarlas con test_metrics.csv.

## Datos

La instantánea incluida contiene 1.827 velas diarias BTC/USDT de Binance, 2020-2024 completos. Se usa close sin ajustes de acciones corporativas. Los timestamps son etiquetas de inicio de vela UTC; el precio de cierre solo está disponible al finalizarla. data/btcusdt_daily.metadata.json registra fuente, solicitudes y hash. La documentación aportada por el usuario es [Gym Trading Env: descarga](https://gym-trading-env.readthedocs.io/en/latest/download.html). Se consultó esa referencia y se utilizó directamente REST de Binance; no se depende de gym-trading-env.

Para descargar una copia nueva sin modificar la instantánea:

```bash
python -m src.download_data --output data/btcusdt_daily_nueva.csv
```

La reproducción de los resultados incluidos no necesita red. Binance puede limitar la disponibilidad por región; en ese caso se conserva el CSV incluido. No se sustituyen fallos de descarga con datos sintéticos.

Para usar el CSV del profesor, prepara columnas `timestamp,close` y, si están disponibles, `open,high,low,volume`. Las fechas deben ser únicas, ascendentes y los precios positivos. Copia configs/default.json, cambia data y elige una salida nueva. El volumen es opcional. No mezcles precios ajustados y sin ajustar. Si recibes un pickle del downloader de Gym Trading Env, conviértelo a CSV solo si procede de una fuente confiable:

```python
import pandas as pd
df = pd.read_pickle("data/archivo_del_downloader.pkl")
df.rename_axis("timestamp").reset_index().to_csv("data/curso.csv", index=False)
```

## Qué se guarda

- config.json, manifest.json, scaler.npz y selection_frozen.json conservan el protocolo.
- models/ contiene seis tablas Q, seis modelos DQN y sus registros de entrenamiento.
- validation_metrics.csv compara recompensas en validación.
- test_metrics.csv conserva cada réplica; test_summary.csv agrega media y desviación estándar.
- validation_*.csv y test_*.csv conservan capital, acciones y costos de cada transición.
- Los PNG muestran capital, drawdown, posiciones, comparación de recompensas y aprendizaje.
- eda_train_summary.csv y notebooks/eda.ipynb permiten inspeccionar entrenamiento.

La selección usa retorno de validación - 0,5 × |drawdown máximo|, promediado entre semillas 7, 21 y 42. Prueba no selecciona algoritmos, recompensas o semillas. Todos los candidatos finales se muestran como auditoría, sin reclasificar la elección congelada.

## Supuestos que afectan la interpretación

El agente observa el cierre y el simulador ejecuta en ese mismo cierre, sin spread o deslizamiento. Es una aproximación educativa optimista. La comisión es 0,10 % del dinero negociado y se deduce antes de asignar pesos. El capital final se valora a mercado sin venta obligatoria, igual para todos. El efectivo no genera intereses. No hay apalancamiento, ventas en corto ni impacto de mercado.

Una sola ventana y tres semillas no demuestran rentabilidad futura. El periodo final incluido ya está observado. Si se hacen cambios después de leerlo, esos resultados pasan a ser exploratorios y se necesita otro periodo intacto para una nueva evaluación final.

## Entrega y exposición

Revisa el informe y completa la reflexión individual con tu experiencia. GUION_EXPOSICION.md propone una presentación de cinco minutos y una guía para el foro. No se recibieron la rúbrica completa ni datos comunes del curso: verifica esos requisitos antes de entregar. No se ha publicado código ni enviado mensajes al foro.
