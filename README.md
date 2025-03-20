﻿# Titanic_surival_project_tl
$$ P(Y=1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n)}} $$
## Mathematical Framework

The logistic regression hypothesis function is given by:

$$
\hat{Y} = \frac{1}{1 + e^{-Z}}
$$

where:

- \( \hat{Y} \) → predicted value  
- \( X \) → Input variable  
- \( w \) → weight  
- \( b \) → bias  

The linear equation is:

$$
Z = wX + b
$$

### **Gradient Descent**
Gradient Descent is an optimization algorithm used for minimizing the loss function in machine learning. It updates the parameters as:

$$
w = w - \alpha \cdot dw
$$

$$
b = b - \alpha \cdot db
$$

### **Learning Rate**
Learning rate \( \alpha \) is a tuning parameter that determines the step size at each iteration while moving toward the minimum of the loss function.

