# Guion para una exposición de cinco minutos

Este guion acompaña el informe de results/main/INFORME.md. Sustituye las referencias a tablas por las cifras de esa ejecución y completa la reflexión con tu experiencia.

## 0:00 a 0:45. Problema y decisión

El agente administra 100.000 USDT y decide qué fracción invertir en BTC. Puede elegir cero, la mitad o todo el patrimonio. La pregunta es si aprende una estrategia que se comporte mejor que comprar y mantener en fechas que no participaron en el entrenamiento. La comparación considera pérdidas y costos, no solo ganancias.

## 0:45 a 1:40. Datos y entorno

Se usan velas diarias BTC/USDT de Binance de 2020 a 2024. Muestra las fechas exactas de las particiones del informe. Explica que validación selecciona candidatos y prueba se reserva para el cierre del experimento. Describe la observación con un ejemplo de tendencia y volatilidad. La comisión es del 0,10 % del dinero negociado y se descuenta antes de invertir. Reconoce el supuesto de ejecutar al mismo cierre observado.

## 1:40 a 2:35. Aprendizaje y recompensa

Se comparan una tabla Q y una red DQN. La primera recompensa es el logaritmo del crecimiento del patrimonio neto. La segunda resta una penalización por pérdidas diarias. Presenta reward_comparison.png y explica qué cambió en retorno, drawdown y tiempo fuera del mercado. No atribuyas una causa que el experimento no permita aislar.

## 2:35 a 3:50. Prueba y baselines

Muestra equity_drawdown.png y la tabla de resultados. Identifica al candidato que fue seleccionado en validación. Compara su retorno con comprar y mantener y menciona el drawdown y las comisiones. Las tres semillas representan variación del aprendizaje sobre el mismo mercado. No son tres periodos financieros independientes.

## 3:50 a 4:35. Fallo o comportamiento inesperado

Muestra positions.png y revisa el peor paso registrado en el informe. Explica qué decisión tomó el agente y cuánto perdió en esa transición. Si su exposición fue casi constante, plantea si aprendió una política sencilla o si no distinguió bien los estados. Expresa como hipótesis lo que no hayas comprobado.

## 4:35 a 5:00. Reflexión

Completa con tu experiencia: qué resultado te sorprendió, qué decisión cambiarías y qué nueva prueba propondrías. Cierra respondiendo la pregunta dentro del alcance de esta ventana histórica. No prometas rentabilidad futura.

## Preguntas del panel

- ¿Cómo evitan usar el futuro? Ventanas causales, escalador ajustado en entrenamiento, sin transiciones que crucen particiones y selección congelada antes de prueba.
- ¿Por qué se cobra al repetir 50 %? La exposición cambia con el precio; volver a 50 % puede requerir rebalancear.
- ¿Por qué DQN? Las acciones son discretas y la red aproxima el valor de cada una. Q-learning permite comparar contra un método clásico.
- ¿Por qué no elegir al mejor en prueba? Eso convertiría la prueba en validación y sesgaría el resultado final.
- ¿Qué falta para una simulación más realista? Ejecutar en la apertura siguiente y añadir spread y deslizamiento con datos consistentes.

## Publicación en el foro

Adjunta el archivo ZIP del proyecto y el informe. Menciona el activo, las fechas, las semillas y el comando de ejecución. Reporta el candidato seleccionado en validación y una limitación concreta. Incorpora tu reflexión personal antes de publicar. Este material no se ha enviado automáticamente a ningún foro.
