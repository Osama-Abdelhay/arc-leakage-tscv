# requires: rsample, dplyr, yardstick, glmnet
suppressPackageStartupMessages({
  library(rsample); library(dplyr); library(yardstick); library(glmnet)
})
set.seed(42)
# toy data
n <- 120
dat <- tibble::tibble(
  date = seq.Date(as.Date("2022-01-01"), by="week", length.out=n),
  x1 = rnorm(n), x2 = rnorm(n, x1, 0.5),
  y  = rbinom(n,1, plogis(0.6*x1 - 0.4*x2))
) %>% arrange(date)

splits <- rolling_origin(dat, initial=80, assess=8, skip=4, cumulative=FALSE)
metrics <- lapply(splits$splits, function(s){
  tr <- analysis(s); te <- assessment(s)
  x_tr <- model.matrix(y~x1+x2, tr)[,-1]; y_tr <- tr$y
  fit <- glmnet(x_tr, y_tr, family="binomial", alpha=0, lambda=0.01)
  p <- predict(fit, newx = model.matrix(y~x1+x2, te)[,-1], type="response")[,1]
  yardstick::roc_auc_vec(te$y, p)
})
cat("Mean AUC:", mean(unlist(metrics)), "\n")
