df <- read.csv('~/Downloads/Auto.csv')
df <- df[df$horsepower != '?', ]
df$horsepower <- as.numeric(df$horsepower)
df <- df[order(df$horsepower), ]
plot(df$horsepower, df$mpg)

library(mgcv)

model <- gam(mpg ~ s(horsepower), data=df)
preds <- predict(model, newdata=df)
lines(df$horsepower, preds)

model <- gam(mpg ~ bs(horsepower), data=df)
preds <- predict(model, newdata=df)
lines(df$horsepower, preds, col='red')

model <- gam(mpg ~ s(horsepower, k=2), data=df)
preds <- predict(model, newdata=df)
lines(df$horsepower, preds, col='red')
