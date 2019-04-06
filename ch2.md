Chapter 2. Statistical Learning
================

# 2.4 Exercises

## Conceptual

1.  For each of parts (a) through (d), indicate whether we would
    generally expect the performance of a flexible statistical learning
    method to be better or worse than an inflexible method. Justify your
    answer.

<!-- end list -->

1.  The sample size n is extremely large, and the number of predictors p
    is small.

A more flexible method should have better performance, because of the
large sample size.

2.  The number of predictors p is extremely large, and the number of
    observations n is small.

A more flexible method should have worse performance, because the small
sample size *n* would lead to a high variance and overfitting.

3.  The relationship between the predictors and response is highly
    non-linear.

A more flexible method should have better performance, because of the
reduced bias.

4.  The variance of the error terms, i.e. \(\sigma^2 = Var(\epsilon)\),
    is extremely high.

Worse performance, because a more flexible method would be more likely
to overfit to the errors in the training data.
