4.6 Lab: Logistic Regression, LDA, QDA, and KNN
================

# 5.3 Lab: Cross-Validation and the Bootstrap

``` r
knitr::opts_chunk$set(warning = FALSE, message = FALSE)
library(tidyverse)
```

    ## -- Attaching packages ------------------------ tidyverse 1.2.1 --

    ## v ggplot2 3.2.0     v purrr   0.3.2
    ## v tibble  2.1.3     v dplyr   0.8.1
    ## v tidyr   0.8.3     v stringr 1.4.0
    ## v readr   1.3.1     v forcats 0.4.0

    ## -- Conflicts --------------------------- tidyverse_conflicts() --
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
library(ISLR)
library(modelr)
library(boot)
set.seed(1)
```

## 5.3.1 The Validation Set Approach

``` r
train_auto <- 
  Auto %>% 
  sample_n(size = 196)

test_auto <- 
  Auto %>% 
  anti_join(train_auto)

lm_auto <- lm(mpg ~ horsepower, data = train_auto)

test_auto %>% 
  add_predictions(lm_auto) %>% 
  mutate(sq_error = (mpg - pred)^2) %>% 
  summarise(mean(sq_error))
```

    ##   mean(sq_error)
    ## 1       23.26601

Trying with polinomial regressions:

``` r
test_eror_poly_lm <- function(grade_poly) {
  lm_n_auto <- lm(mpg ~ poly(horsepower, grade_poly), data = train_auto)

test_auto %>% 
  add_predictions(lm_n_auto) %>% 
  mutate(sq_error = (mpg - pred)^2) %>% 
  summarise(mean(sq_error))
}
```

``` r
test_eror_poly_lm(2)
```

    ##   mean(sq_error)
    ## 1       18.71646

``` r
test_eror_poly_lm(3)
```

    ##   mean(sq_error)
    ## 1       18.79401

Using a different test/train split (to see how the test error changes):

``` r
set.seed(2)

train_auto <- 
  Auto %>% 
  sample_n(size = 196)

test_auto <- 
  Auto %>% 
  anti_join(train_auto)
```

``` r
test_eror_poly_lm(1)
```

    ##   mean(sq_error)
    ## 1       25.72651

``` r
test_eror_poly_lm(2)
```

    ##   mean(sq_error)
    ## 1       20.43036

``` r
test_eror_poly_lm(3)
```

    ##   mean(sq_error)
    ## 1       20.38533

## 5.3.2 Leave-One-Out Cross-Validation

Estimating the test error with LOOCV in linear regression:

``` r
glm_auto <- glm(mpg ~ horsepower, data = Auto)

cv_err <- cv.glm(Auto, glm_auto)

cv_err[["delta"]]
```

    ## [1] 24.23151 24.23114

Repeating for more complex polynomial fits:

``` r
loocv_error_poly <- function(n){
  glm_auto <- glm(mpg ~ poly(horsepower, n), data = Auto)

  cv_err <- cv.glm(Auto, glm_auto)
  
  cv_err[["delta"]][[1]]
}

map_dbl(1:5, loocv_error_poly)
```

    ## [1] 24.23151 19.24821 19.33498 19.42443 19.03321

We see a sharp decrease from linear fit to quadratic fit, but not so
much in cubic fit and beyond.

## 5.3.3 k-Fold Cross-Validation

``` r
set.seed(17)
k10_error_poly <- function(n){
  glm_auto <- glm(mpg ~ poly(horsepower, n), data = Auto)

  cv_err_10 <- cv.glm(Auto, glm_auto, K = 10)
  
  cv_err_10[["delta"]][[1]]
}

map_dbl(1:10, k10_error_poly)
```

    ##  [1] 24.27207 19.26909 19.34805 19.29496 19.03198 18.89781 19.12061
    ##  [8] 19.14666 18.87013 20.95520

Note: the two numbers associated with `delta` are essentially the same
when LOOCV is performed. When we instead perform k-fold CV, then the two
numbers associated with `delta` differ slightly. The first is the
standard k-fold CV estimate, and the second is a bias corrected version.

## 5.3.4 The Bootstrap

First we create a function to compute the alpha statistic:

``` r
alpha_fn <- function (data, index){
  X <- data$X[index]
  Y <- data$Y[index]
  
  (var(Y)-cov(X,Y))/(var(X)+var(Y) -2*cov(X,Y))
}
```

Then we perform bootstrap with the `boot` function:

``` r
boot(Portfolio, alpha_fn, R=1000)
```

    ## 
    ## ORDINARY NONPARAMETRIC BOOTSTRAP
    ## 
    ## 
    ## Call:
    ## boot(data = Portfolio, statistic = alpha_fn, R = 1000)
    ## 
    ## 
    ## Bootstrap Statistics :
    ##      original     bias    std. error
    ## t1* 0.5758321 0.00705678  0.09050198

Comparing the standard errors of coefficients estimated by bootstrap
vs. estimated with `lm()`

``` r
coefs_boot <- function (data, index) {
  coef(lm(mpg∼horsepower , data = data , subset = index))
}

coefs_boot(Auto, 1:392)
```

    ## (Intercept)  horsepower 
    ##  39.9358610  -0.1578447

``` r
coefs_boot(Auto, sample(1:392, 392, replace = TRUE))
```

    ## (Intercept)  horsepower 
    ##  39.3577450  -0.1539671

``` r
coefs_boot(Auto, sample(1:392, 392, replace = TRUE))
```

    ## (Intercept)  horsepower 
    ##   40.436161   -0.165136

``` r
boot(Auto, coefs_boot, R = 10000)
```

    ## 
    ## ORDINARY NONPARAMETRIC BOOTSTRAP
    ## 
    ## 
    ## Call:
    ## boot(data = Auto, statistic = coefs_boot, R = 10000)
    ## 
    ## 
    ## Bootstrap Statistics :
    ##       original        bias    std. error
    ## t1* 39.9358610  0.0278211233 0.858649495
    ## t2* -0.1578447 -0.0003406418 0.007443214

``` r
lm(mpg ∼ horsepower, data = Auto) %>% summary()
```

    ## 
    ## Call:
    ## lm(formula = mpg ~ horsepower, data = Auto)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -13.5710  -3.2592  -0.3435   2.7630  16.9240 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) 39.935861   0.717499   55.66   <2e-16 ***
    ## horsepower  -0.157845   0.006446  -24.49   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 4.906 on 390 degrees of freedom
    ## Multiple R-squared:  0.6059, Adjusted R-squared:  0.6049 
    ## F-statistic: 599.7 on 1 and 390 DF,  p-value: < 2.2e-16

The bootstrap estimate for coeffcients std. errors is slightly higher
than the `lm()` estimate. In fact, the bootstrap estimate is more
accurate, because it doesn’t rely on the linear model asumptions.

Now let’s see the difference in estimate when using a quadratic model
(which better fits this data.)

``` r
coefs_boot_lm2 <- function (data, index) {
  coef(lm(mpg ∼ horsepower + I(horsepower^2), data = data , subset = index))
}

set.seed(1)

boot(Auto, coefs_boot_lm2, R = 1000)
```

    ## 
    ## ORDINARY NONPARAMETRIC BOOTSTRAP
    ## 
    ## 
    ## Call:
    ## boot(data = Auto, statistic = coefs_boot_lm2, R = 1000)
    ## 
    ## 
    ## Bootstrap Statistics :
    ##         original        bias     std. error
    ## t1* 56.900099702  3.511640e-02 2.0300222526
    ## t2* -0.466189630 -7.080834e-04 0.0324241984
    ## t3*  0.001230536  2.840324e-06 0.0001172164

``` r
lm(mpg ∼ horsepower + I(horsepower^2), data = Auto) %>% summary()
```

    ## 
    ## Call:
    ## lm(formula = mpg ~ horsepower + I(horsepower^2), data = Auto)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -14.7135  -2.5943  -0.0859   2.2868  15.8961 
    ## 
    ## Coefficients:
    ##                   Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)     56.9000997  1.8004268   31.60   <2e-16 ***
    ## horsepower      -0.4661896  0.0311246  -14.98   <2e-16 ***
    ## I(horsepower^2)  0.0012305  0.0001221   10.08   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 4.374 on 389 degrees of freedom
    ## Multiple R-squared:  0.6876, Adjusted R-squared:  0.686 
    ## F-statistic:   428 on 2 and 389 DF,  p-value: < 2.2e-16

Since the cuadratic model is closer to the true structure of the data,
the difference between both estimates for standard errors is smaller.
