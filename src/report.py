"""Informe en español con valores calculados, sin resultados inventados."""
from pathlib import Path
import pandas as pd


def table(frame, columns):
    header = "| " + " | ".join(columns) + " |\n|" + "|".join(["---"]*len(columns)) + "|\n"
    return header + "\n".join("| " + " | ".join(str(row[c]) for c in columns) + " |" for _, row in frame.iterrows())


def write_report(out, manifest, cfg, validation, results, selected):
    from .data import prepare
    data, _, periods, _, _ = prepare(cfg["data"], cfg)
    training = data.iloc[:periods["train"][1]+1]
    low = training.loc[training.return_1.idxmin()]
    high_vol = training.loc[training.volatility_20.idxmax()]
    low_vol = training.loc[training.volatility_20.idxmin()]
    asset_drawdown = (training.close/training.close.cummax()-1).min()
    volume_sentence = (f"El volumen diario mediano fue {training.volume.median():,.2f} BTC. "
                       if 'volume' in training else "El CSV no contiene volumen. ")
    winner = results[(results.method == selected["method"]) & (results.reward == selected["reward"])]
    buy = results[results.method == "buy_hold"].iloc[0]
    cash = results[results.method == "cash"].iloc[0]
    rows = []
    labels = {"qlearning": "Q-learning", "dqn": "DQN", "buy_hold": "Comprar y mantener",
              "random": "Aleatorio", "cash": "Efectivo"}
    for (kind, reward), group in results.groupby(["method", "reward"]):
        rows.append({"Estrategia": labels[kind]+(" / "+reward if reward != "baseline" else ""),
                     "Retorno medio": f"{group['return'].mean():.2%}",
                     "DE retorno": f"{group['return'].std():.2%}" if len(group)>1 else "No aplica",
                     "Drawdown medio": f"{group.max_drawdown.mean():.2%}",
                     "Costos medios": f"{group.costs.mean():,.2f}",
                     "Operaciones medias": f"{group.trades.mean():.1f}",
                     "Réplicas": len(group)})
    rows = pd.DataFrame(rows)
    vr = []
    for (kind, reward), group in validation.groupby(["method", "reward"]):
        vr.append({"Método": labels[kind], "Recompensa": reward,
                   "Retorno": f"{group['return'].mean():.2%}",
                   "Drawdown": f"{group.max_drawdown.mean():.2%}",
                   "Costos": f"{group.costs.mean():,.2f}",
                   "Tiempo en efectivo": f"{group.cash_fraction.mean():.2%}",
                   "Puntaje": f"{group.score.mean():.6f}"})
    vr = pd.DataFrame(vr)
    splits = "\n".join(f"- {k}: {v['start'][:10]} a {v['end'][:10]}, {v['transitions']} transiciones."
                       for k, v in manifest["splits"].items())
    gain = winner['return'].mean()-buy['return']
    verdict = ("En esta ventana, el promedio no respalda una mejora del agente seleccionado sobre comprar y mantener."
               if gain <= 0 else "En esta ventana, el promedio de retorno fue mayor que comprar y mantener; esto no demuestra una ventaja en otros periodos.")
    q_log_test = results[(results.method == 'qlearning') & (results.reward == 'log')]
    q_log_val = validation[(validation.method == 'qlearning') & (validation.reward == 'log')]
    beats = int((winner['return'] > buy['return']).sum())
    obs = []
    for _, row in winner.iterrows():
        name = f"{row.method}_{row.reward}_{int(row.seed)}"
        curve = pd.read_csv(out/f"test_{name}.csv").iloc[1:]
        worst = curve.loc[curve.net_return.idxmin()]
        obs.append(f"Semilla {int(row.seed)}: {row.cash_fraction:.1%} de decisiones con exposición cero, "
                   f"exposición objetivo media {row.mean_target_weight:.1%}. Peor transición: "
                   f"{str(worst.decision_timestamp)[:10]} a {str(worst.timestamp)[:10]}, "
                   f"acción {int(worst.action)}, retorno neto {worst.net_return:.2%}.")
    log_v = validation[(validation.method == selected['method']) & (validation.reward == 'log')]
    risk_v = validation[(validation.method == selected['method']) & (validation.reward == 'risk')]
    text = f"""# Aprendizaje por refuerzo para decidir la exposición a BTC/USDT

## Alcance y pregunta de investigación

¿Puede un agente aprender una política que supere a comprar y mantener fuera del entrenamiento, después de pagar costos y considerar las pérdidas? Se implementaron Q-learning tabular y DQN con acciones de 0 %, 50 % y 100 %. El capital inicial fue de 100.000 unidades de cotización, aquí USDT. No se realiza conversión a pesos colombianos.

El candidato elegido en validación fue {labels[selected['method']]} con recompensa {selected['reward']}. En prueba obtuvo un retorno medio de {winner['return'].mean():.2%}; comprar y mantener obtuvo {buy['return']:.2%}. La diferencia fue {gain*100:.2f} puntos porcentuales. El drawdown medio del agente fue {winner.max_drawdown.mean():.2%}, frente a {buy.max_drawdown:.2%} del baseline. Estos valores describen una ventana histórica; no prueban superioridad en otros periodos.

## Datos y separación temporal

Se utilizaron 1.827 velas diarias de BTC/USDT del mercado spot de Binance, del 1 de enero de 2020 al 31 de diciembre de 2024. La página de Gym Trading Env facilitada para el trabajo documenta la descarga de criptomonedas con CCXT. Este proyecto usa el endpoint público REST de Binance para conservar una instantánea CSV sin añadir ese paquete. La referencia orienta la fuente; no es un archivo de precios entregado por el profesor.

Cada timestamp identifica el comienzo de la vela UTC. El cierre y los indicadores de esa fila se consideran conocidos al terminar el día. Las fechas del informe son etiquetas de velas, no horas de disponibilidad intradía. El volumen corresponde a BTC. El CSV y sus metadatos están incluidos; repetir el experimento no requiere red.

Tras descartar las primeras 20 observaciones de calentamiento, la división cronológica es 60 % / 20 % / 20 %:

{splits}

Las transiciones entre particiones se excluyen: cada periodo inicia con efectivo y termina en su último precio. Una media móvil de validación o prueba puede usar precios anteriores ya conocidos. Ninguna transición de entrenamiento utiliza un precio de validación. La media y la desviación del escalamiento se ajustan exclusivamente en entrenamiento; después quedan fijas. No se barajan las filas.

SHA-256 del CSV: `{manifest['sha256']}`.

## Formulación del proceso de decisión

La observación tiene siete componentes. El retorno de una sesión resume el movimiento reciente. Los retornos de 5 y 20 sesiones describen tendencias de distintas duraciones. La diferencia relativa entre precio y media de 20 sesiones mide separación frente al nivel reciente. La desviación de los retornos de 20 sesiones aproxima la variabilidad. El peso actual informa cuánto capital ya está invertido y el drawdown indica la pérdida respecto al máximo del patrimonio.

Los cinco indicadores de mercado se estandarizan con el entrenamiento y se limitan al intervalo [-10, 10]. El peso pertenece a [0, 1] y el drawdown a [-1, 0]. Las ventanas usan únicamente precios hasta t, sin centrado ni relleno hacia atrás. La observación es una representación parcial del mercado. No se demuestra la propiedad de Markov: variables externas y regímenes ocultos pueden afectar al precio siguiente.

La acción a pertenece a {{0, 1, 2}} y determina el peso objetivo w en {{0, 0,5, 1}} después de pagar la comisión. Esta discretización permite comparar Q-learning y DQN. Se admiten participaciones fraccionarias. No hay ventas en corto ni apalancamiento. Tampoco hay posiciones intermedias distintas de 50 %, órdenes límite, intereses sobre efectivo o impacto de mercado.

## Entorno y contabilidad

Al finalizar la vela t, el agente observa el estado, solicita una exposición y el simulador ejecuta al precio de cierre t. Luego valora la posición al cierre t+1 y entrega la recompensa. Se trata de una ejecución idealizada al mismo cierre observado. No usa el cierre t+1 para elegir la acción, pero supone una liquidez y un precio de ejecución que no están garantizados en una operación real. Evaluar con ejecución en la apertura siguiente y datos consistentes de OHLC sería una extensión necesaria antes de interpretar los resultados como operables.

Sea V el patrimonio antes de operar, B el valor del activo antes del rebalanceo, X el valor del activo tras operar y c = {cfg['cost_rate']:.4f}. Se resuelve X = w[V - c|X-B|]. El costo monetario es C = c|X-B| y el efectivo restante es V-C-X. En compras, X = w(V+cB)/(1+wc); en ventas, X = w(V-cB)/(1-wc). Esto evita invertir todo el patrimonio y pagar la comisión con un saldo negativo.

Si R = p(t+1)/p(t)-1, entonces V(t+1) = V-C+XR. El peso deriva con el precio: mantener la misma acción 50 % puede exigir operar de nuevo. El costo depende del volumen monetario negociado, no de si cambió la etiqueta de acción. Comprar y mantener paga su compra inicial y conserva el activo. Todos comienzan en efectivo, pagan la misma tasa y se valoran a mercado al final, sin venta terminal obligatoria. El drawdown incluye el capital inicial.

La serie de mercado es exógena: las decisiones del agente no afectan el precio. El episodio termina al agotar las transiciones del periodo. El simulador implementa reset y step de Gymnasium y supera el comprobador de Stable-Baselines3.

## Diseño e iteración de la recompensa

La versión inicial es r1 = 100 log[V(t+1)/V(t)]. Incluye las comisiones dentro del patrimonio. La suma de log-retornos equivale al logaritmo del crecimiento acumulado, pero no penaliza por separado la exposición a pérdidas diarias.

La versión revisada es r2 = 100 {{log[V(t+1)/V(t)] - lambda × min(g, 0)^2}}, con g = V(t+1)/V(t)-1 y lambda = {cfg['risk_lambda']:.1f}. El segundo término penaliza pérdidas netas grandes. El factor 100 es fijo en ambas versiones y solo cambia la escala. La penalización adicional cambia el objetivo económico y puede favorecer quedarse en efectivo.

Se entrenó primero la versión log y después la versión risk de cada algoritmo. La comparación utiliza validación; lambda y el presupuesto de entrenamiento se fijaron antes de abrir la prueba. Para {labels[selected['method']]}, log presentó retorno medio {log_v['return'].mean():.2%}, drawdown {log_v.max_drawdown.mean():.2%} y tiempo en efectivo {log_v.cash_fraction.mean():.2%}. Risk presentó retorno {risk_v['return'].mean():.2%}, drawdown {risk_v.max_drawdown.mean():.2%} y tiempo en efectivo {risk_v.cash_fraction.mean():.2%}. Esos datos permiten revisar si el cambio redujo pérdidas o si restringió demasiado la participación.

La hipótesis de la segunda versión fue controlar las pérdidas diarias; no se afirma que su coeficiente sea óptimo. Las diferencias entre versiones son evidencia descriptiva de esta ejecución, no una prueba causal aislada de la penalización.

{table(vr, list(vr.columns))}

![Comparación en validación](reward_comparison.png)

Las barras de error representan una desviación estándar muestral del puntaje entre semillas. No son intervalos de confianza sobre mercados futuros.

Un puntaje alto de recompensa no basta para preferir una estrategia: quedarse fuera del mercado puede evitar penalizaciones y perder una subida. Por eso se reportan métricas monetarias, exposición y operaciones. No se identificó una explotación contable del entorno en las pruebas; eso no descarta otros comportamientos no deseados.

## Algoritmos y entrenamiento

Q-learning utiliza una tabla de 243 estados y 3 acciones. Discretiza momentum de 5 y 20 días y volatilidad estandarizados con cortes -0,5 y 0,5; el peso con 0,25 y 0,75; y el drawdown con -0,15 y -0,05. Descarta dos indicadores del estado completo para contener el número de celdas. Usa alpha = 0,1, gamma = 0,99 y {cfg['q_episodes']} episodios completos. Epsilon baja linealmente de 1 a 0,05 en el primer 80 % de episodios. En evaluación se elige argmax; los empates favorecen la acción cero, lo cual puede sesgar estados sin visitas hacia efectivo.

DQN aproxima Q(s,a) con una red de siete entradas, dos capas ocultas de 64 unidades con ReLU y tres salidas. Usa CPU y una hebra de PyTorch. Hiperparámetros: learning rate {cfg['dqn']['learning_rate']}, batch {cfg['dqn']['batch_size']}, gamma {cfg['dqn']['gamma']}, replay buffer {cfg['dqn']['buffer_size']}, inicio del aprendizaje en el paso {cfg['dqn']['learning_starts']}, un gradiente cada {cfg['dqn']['train_freq']} pasos y copia de la red objetivo cada {cfg['dqn']['target_update_interval']} pasos. La exploración epsilon baja de 1 a 0,05 durante el primer 40 % de {cfg['timesteps']} pasos. Se detiene al agotar ese presupuesto, sin parada temprana basada en prueba. Se guarda el último modelo.

Semillas de entrenamiento y del baseline aleatorio: {cfg['seeds']}. La política de evaluación DQN es determinista. Las semillas replican la inicialización y la exploración, no crean nuevos mercados independientes. El número de interacciones de Q-learning es {cfg['q_episodes']} × {manifest['splits']['train']['transitions']}; el de DQN es {cfg['timesteps']}. La comparación no iguala el presupuesto de cómputo. Repetir episodios reutiliza el mismo recorrido histórico y puede producir memorización.

![Curvas de entrenamiento](learning_curves.png)

Las curvas muestran una media móvil dentro de cada entrenamiento. Sus alturas no comparan directamente objetivos distintos: risk resta una penalización que log no contiene. Mejorar la recompensa de entrenamiento tampoco demuestra que el agente generalice, como muestra la diferencia entre validación y prueba.

## Protocolo de evaluación

Se entrenan cuatro candidatos: cada algoritmo con cada recompensa. La regla de selección, fijada antes de prueba, es la media entre semillas de retorno de validación menos 0,5 veces el valor absoluto del drawdown máximo de validación. Este coeficiente expresa una preferencia explícita por limitar pérdidas; otra preferencia podría elegir otro candidato.

El archivo selection_frozen.json conserva la selección antes de ejecutar los recorridos finales. Se elige una combinación de algoritmo y recompensa, sin escoger la semilla ganadora. No se reentrena con validación. Se evalúan sus tres réplicas en prueba y también se muestran los otros candidatos como auditoría descriptiva. Sus resultados finales no cambian la selección ni los parámetros.

Los baselines son comprar y mantener, aleatorio uniforme entre las tres acciones y efectivo. Comparten fechas, capital inicial y comisiones. Comprar y mantener y efectivo tienen un solo recorrido porque no dependen de semillas. Se reporta la media y la desviación estándar muestral entre semillas; esta dispersión no es un intervalo de confianza sobre mercados futuros. No se realizan pruebas de significancia con esas tres réplicas como si fueran trayectorias independientes.

## Resultados fuera de muestra

{table(rows, list(rows.columns))}

Los costos están expresados en USDT nominales acumulados. Un drawdown de -20 % corresponde a una caída del 20 % desde un máximo. La desviación entre semillas puede ser cero si todas convergen a la misma política. La recompensa total no se compara directamente entre log y risk, pues miden objetivos distintos.

![Capital y drawdown](equity_drawdown.png)

El agente seleccionado pagó {winner.costs.mean():,.2f} USDT de costos medios, frente a {buy.costs:,.2f} de comprar y mantener. La comparación debe considerar conjuntamente retorno y drawdown: una mejora de un indicador con deterioro del otro no implica dominar al baseline.

En {beats} de las {len(winner)} semillas el agente seleccionado superó el retorno de comprar y mantener. Sus retornos variaron de {winner['return'].min():.2%} a {winner['return'].max():.2%}. Esa amplitud muestra que la conclusión cambia si se publica solo una semilla. La diferencia media fue {gain*100:.2f} puntos porcentuales. {verdict}

El cambio de validación a prueba también produjo un fallo de generalización observable: Q-learning con recompensa log pasó de un retorno medio de {q_log_val['return'].mean():.2%} a {q_log_test['return'].mean():.2%}. Esta comparación no mantiene fijo el mercado, por lo que no permite identificar la causa. Sí muestra que un resultado favorable en validación no asegura que la política conserve su desempeño después.

![Posiciones del agente](positions.png)

### Comportamiento observado y anomalías

{' '.join(obs)}

Como diagnóstico reproducible, cada CSV conserva la decisión, el retorno siguiente, la comisión y la exposición. El peor paso anterior permite revisar una pérdida concreta. Una política con exposición casi constante también merece revisión: puede haber aprendido una solución sencilla o haber dejado de distinguir estados. La causa no puede atribuirse a una sola variable sin nuevos experimentos realizados sobre entrenamiento y validación.

## Análisis exploratorio

El cuaderno notebooks/eda.ipynb analiza solo entrenamiento en su configuración inicial. Incluye precio, distribución de retornos, volatilidad móvil, drawdown del activo, volumen e indicadores. Se reserva el EDA completo para la descripción posterior al cierre del experimento. La siguiente figura reúne todo el historial y no se utilizó para seleccionar modelos.

![Historial completo, descripción posterior](eda.png)

Los retornos y la volatilidad se construyen con ventanas móviles para que el estado no dependa del nivel absoluto de BTC. La exposición y el drawdown añaden información del portafolio que el precio no describe. El volumen se analiza, pero se excluye del estado para mantener una representación pequeña. El archivo eda_train_summary.csv contiene estadísticas calculadas únicamente con entrenamiento.

En entrenamiento, el cierre mínimo fue {training.close.min():,.2f} USDT y el máximo {training.close.max():,.2f} USDT. La mayor caída diaria fue {low.return_1:.2%}, en la vela {str(low.timestamp)[:10]}. El drawdown máximo del precio fue {asset_drawdown:.2%}. Por eso una recompensa basada solo en crecimiento no describe toda la experiencia de pérdidas del inversionista.

La volatilidad móvil de 20 días alcanzó {high_vol.volatility_20:.2%} diario el {str(high_vol.timestamp)[:10]} y bajó hasta {low_vol.volatility_20:.2%} el {str(low_vol.timestamp)[:10]}. Estos cambios justifican observar volatilidad en vez de asumir un riesgo constante. {volume_sentence}La correlación de momentum a 20 días con la distancia a su media fue {training.momentum_20.corr(training.gap_sma20):.3f}, lo que indica información parcialmente redundante.

## Limitaciones y respuesta a la pregunta

En esta ejecución, la diferencia media de retorno del candidato seleccionado respecto a comprar y mantener fue {gain*100:.2f} puntos porcentuales. El resultado debe leerse junto con el riesgo y los costos de la tabla. No autoriza a afirmar que RL sea superior en mercados financieros en general. El baseline efectivo obtuvo {cash['return']:.2%}, lo que ayuda a detectar si operar destruyó capital.

El estudio usa un activo y una sola división temporal. El periodo final puede favorecer un régimen particular; una evaluación walk-forward en nuevos periodos ayudaría a estimar estabilidad. No incluye spread, deslizamiento, impuestos o restricciones de tamaño de orden. El efectivo no gana intereses y USDT se trata como unidad estable. La ejecución al mismo cierre observado es una aproximación optimista. Los datos de una bolsa no representan todos los mercados.

No se recibieron la rúbrica completa ni un CSV común del profesor. Antes de entregar, debe comprobarse si se exige otro activo o periodo. En ese caso, se cambia el CSV y se registra un experimento nuevo; el periodo final de este experimento ya fue observado y no debe reutilizarse para ajustar parámetros.

## Reflexión sobre el experimento y guía para la reflexión individual

El contraste que merece mayor revisión es la variación entre semillas: {winner['return'].min():.2%} frente a {winner['return'].max():.2%} para el mismo algoritmo, datos y recompensa. Escoger el mejor recorrido habría contado una historia distinta a la media de las réplicas. La reflexión personal puede explicar si esta diferencia coincidió con lo que se esperaba antes de ejecutar.

La separación de validación y prueba cambió la interpretación del resultado. El candidato se eligió por validación y conservó esa condición aun cuando su media final quedó por debajo de comprar y mantener. La recompensa de riesgo tampoco aseguró una mejora: los resultados de validación muestran que penalizar pérdidas no basta para controlar todos los cambios de exposición. No se midió de forma aislada qué decisión tuvo el mayor efecto causal.

Una repetición podría usar varios cortes walk-forward con presupuestos de interacción equivalentes entre métodos. Antes de entrenar habría que fijar periodos, métricas y regla de selección. La ejecución en la apertura siguiente permitiría estudiar cuánto depende el resultado del supuesto de operar al mismo cierre observado.

No está completamente explicada la diferencia de exposición entre semillas DQN. Los registros permiten localizar cuándo ocurre, pero no identificar por sí solos si nace de la inicialización, la exploración o la cobertura de estados. Una comprobación sería fijar una de esas fuentes de variación en entrenamiento y cambiar otra, manteniendo intacta una nueva prueba futura.

La aplicación financiera añade una dificultad que un entorno didáctico puede ocultar: repetir episodios reproduce la misma historia y no genera mercados nuevos. El agente puede mejorar su recompensa sobre ese recorrido sin aprender una política estable fuera de él. Las comisiones convierten parte de la actividad aparente en pérdida de capital y obligan a revisar el comportamiento, no solo la recompensa.

Para la entrega individual, estas observaciones deben adaptarse a la experiencia de quien presenta. Conviene responder con palabras propias qué resultado sorprendió, qué decisión cambiaría y qué comportamiento sigue sin comprender; no atribuirse pruebas o vivencias que no se realizaron.

## Reproducción y archivos de evidencia

Desde la carpeta del proyecto: python -m pip install -r requirements.txt; python -m unittest discover -s tests -v; python -m src.train --config configs/default.json --output results/replica_01. Los comandos detallados están en README.md. No se necesita descargar datos para reproducir esta instantánea.

Cada ejecución guarda modelos, escalador, configuración, versiones, hash del dataset y métricas por semilla. Los archivos test_*.csv permiten recalcular las métricas y reconstruir los gráficos. Las curvas de aprendizaje DQN están en models/*_monitor.csv; las de Q-learning en learning_*.csv. Los errores de red o dependencias no se sustituyen por datos inventados.

## Referencias

- Sutton, R. S., y Barto, A. G. (2018). Reinforcement Learning: An Introduction, segunda edición. https://incompleteideas.net/book/the-book-2nd.html
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. https://doi.org/10.1038/nature14236
- Stable-Baselines3. Documentación DQN. https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
- Gymnasium. API Env. https://gymnasium.farama.org/api/env/
- Gym Trading Env. Download market data. https://gym-trading-env.readthedocs.io/en/latest/download.html
- Binance. Documentación de velas del API spot. https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#klinecandlestick-data
"""
    (out/"INFORME.md").write_text(text, encoding="utf-8")
