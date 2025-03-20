# Titanic_survival_project_tl
 
# Why i choose the project 
I just had zero idea about any of the given project but titanic survival seemed interesting so i choose to learn that

# Key learning from project
confessiion: (I didn't do it on my full own i used chatgpt and youtube channel how to make this project but even though now i atleast know the concept which i used)
1.we should first properly clean the dataset and we should choose the input feature properly
2.rich and women will be given importane during catastrophe
3.learnt how github works 


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


# Challenges faced during working on this project
1.first i thought knowing basic python (for loop etc) is more than enough for my project  but after working on it only i got to know what all the things are basic
2.i was struggling to make mathmetical framework in readme but somehow managed to make it
3. learning git and pushing my file to it was the hardest challenge i faced , everytime i tried  i got error only but after so much help from chatgpt i done it

# Setup 
1.Clone the repository 
2.go to the respected directory 
3.makesure install the listed dependencies installed 

-numpy 
-pandas
-sklearn

