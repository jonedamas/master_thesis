
url <- "C://Users/joneh/master_thesis/data/time_series/YahooFinance/CL=F_20years.csv"

data <- read.csv(url)

print(head(data))

model <- lm(Adj.Close ~ Volume, data = data)

plot(
    data$Volume, 
    data$Adj.Close, 
    xlab = "Volume", 
    ylab = "Adj.Close", 
    main = "Adj.Close vs Volume", 
    col = "blue", 
    pch = 19
)

print(summary(model))