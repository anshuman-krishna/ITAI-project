# Feature importance — tfidf+logistic_regression

Top weighted tokens from the fitted linear model's coefficients (no SHAP). Positive
weights push a text toward **ai**, negative toward **human**.

## Pushes toward ai

| token | weight |
| --- | --- |
| nationally | +0.4094 |
| votes | +0.4007 |
| larger | +0.2640 |
| trump | +0.2631 |
| bound | +0.2442 |
| costs | +0.2348 |
| candidate | +0.2324 |
| granted | +0.2293 |
| individual | +0.1993 |
| win | +0.1974 |
| state | +0.1913 |
| furthermore | +0.1676 |
| analyze | +0.1636 |
| unless | +0.1606 |
| won | +0.1571 |
| services | +0.1558 |
| tend | +0.1485 |
| metropolitan | +0.1482 |
| reflect | +0.1459 |
| increased | +0.1424 |

## Pushes toward human

| token | weight |
| --- | --- |
| cars | -0.2039 |
| people | -0.1174 |
| car | -0.1004 |
| source | -0.0850 |
| usage | -0.0831 |
| driving | -0.0803 |
| get | -0.0770 |
| less | -0.0768 |
| air | -0.0746 |
| smog | -0.0740 |
| candidates | -0.0736 |
| voters | -0.0720 |
| day | -0.0702 |
| voting | -0.0673 |
| world | -0.0670 |
| dont | -0.0634 |
| like | -0.0616 |
| unfair | -0.0613 |
| percent | -0.0603 |
| may | -0.0572 |
