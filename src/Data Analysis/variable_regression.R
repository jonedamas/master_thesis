
url <- "C://Users/joneh/master_thesis/data/regression/combined_df_NYT.csv"

data <- read.csv(url)

print(head(data))

model <- lm(GARCH ~ SV + polarity, data = data)

print(summary(model))
