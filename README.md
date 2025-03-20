# Titanic_surival_project_tl

## Mathematical Framework

The logistic regression hypothesis function is given by:
$$
\hat{Y} = \frac{1}{1 + e^{-Z}}
$$

where:

- \$$( \hat{Y} \)$$ → predicted value  
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
Learning rate is a tuning parameter that determines the step size at each iteration while moving toward the minimum of the loss function.

### **Derivatives:**

The gradients for weight \( w \) and bias \( b \) are calculated as:

$$
dw = \frac{1}{m} \sum ( \hat{Y} - Y ) \cdot X
$$

$$
db = \frac{1}{m} \sum ( \hat{Y} - Y )
$$

where:

- \( m \) → number of training examples  
- \$$( \hat{Y} \)$$ → predicted value  (can range from 0 to 1)
- \( Y \) → actual value  
- \( X \) → input variable





