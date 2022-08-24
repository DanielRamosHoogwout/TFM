#DM test
setwd("C:/Users/Daniel/Desktop/TFM")
library(forecast)

error_30 <- read.csv("error_30.csv")
error_30 <- error_30[['Data']]

error_60 <- read.csv("error_60.csv")
error_60 <- error_60[['Data']]

error_90 <- read.csv("error_90.csv")
error_90 <- error_90[['Data']]

error_120 <- read.csv("error_120.csv")
error_120 <- error_120[['Data']]

## El benchmark o modelo de comparación es el 90. Así que todos los demás
## se comparan con este modelo. Además, la hipótesis alternativa es que
## el modelo benchmark 90 predice mejor y, por ende, obtiene errores de pre-
## dicción menores ("less") que los del modelo de comparación. Así que 
## el código correcto sería:

dm_90_30 <- dm.test(error_90, error_30, alternative = c("less"))
print(dm_90_30[c("statistic","p.value")])

dm_90_60 <- dm.test(error_90, error_60, alternative = c("less"))
print(dm_90_60[c("statistic","p.value")])

dm_90_120 <- dm.test(error_90, error_120, alternative = c("less"))
print(dm_90_120[c("statistic","p.value")])

## No hace falta hacer comparaciones entre los demás modelos porque no los
## has elegido en los análisis anteriores.