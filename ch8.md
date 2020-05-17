8.4 Exercises
================

## Conceptual

(1) Draw an example (of your own invention) of a partition of two
dimensional feature space that could result from recursive binary
splitting. Your example should contain at least six regions. Draw a
decision tree corresponding to this partition. Be sure to label all
aspects of your figures, including the regions \(R_1, R_2, ...\),the
cutpoints \(t_1,t_2, ...\), and so forth.

![](ch8_files/exc1_1.png)

![](ch8_files/exc1_2.png)

(2) It is mentioned in Section 8.2.3 that boosting using depth-one trees
(or stumps) leads to an additive model: that is, a model of the form

![](ch8_files/exc2_instr.png)

Explain why this is the case. You can begin with (8.12) in Algorithm
8.2.

A: The key to understand why a boosting based on stumps can be expressed
as an additive model is that in the original formula of boosting:

![](ch8_files/exc2_1.png)

Each tree is a function of the whole vector of predictors (\(X\)) but in
this case, each tree (stump) is a function of just one predictor:

![](ch8_files/exc2_2.png)

So, for all the \(B\) stumps that are added in the boosting formula, we
can assign them to “sets” or “groups” based on the predictor which each
of them uses:

![](ch8_files/exc2_3.png)
