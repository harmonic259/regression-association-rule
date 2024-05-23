# regression-association-rule

This project contains 2 main part:
1. Regression algorithms
2. Association rules

I wrote the regression algorithms part from the scratch (just using numpy functions).

## Regression
The following regression algorithms was implemented in this part:
<details>
<summary>
Algorithms
</summary>
  
1. Linear rgression
2. Polynomial regression
3. Ridge regression
4. Lasso regression
5. Elastic net regression
6. Gradient boosting regression
   
</details>

The first 5 algorithms are easy to read and understand but for the 6th one, I implemented a very simple version of [Gradient boosing for regression](https://en.wikipedia.org/wiki/Gradient_boosting#Algorithm)
:
<details>
<summary>
Code
</summary>
  
```python
class XGboostReg:
    def __init__(self, steps=5):
        self.steps = steps
        self.lin_reg_params = []
    def fit(self, X, y):
        n = np.size(y)
        l = np.ones(n) * sum([yi for yi in y]) / n 
        for _ in range(self.steps):
            resid_y = np.array([(y[i] - l[i]) for i in range(n)]) * 2 / n
            self.lin_reg_fit(X, resid_y)
            l = self.pred(X)
            
            
    def lin_reg_fit(self, X, y):
        self.lin_reg_params.append(np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y)))
    def lin_reg_pred(self, X):
        return np.matmul(X, self.lin_reg_params)
    def pred(self, X):
        n = np.shape(X)[0]
        y = np.zeros(n)
        y = sum([np.matmul(X, self.lin_reg_params[j]) for j in range(len(self.lin_reg_params))])
        return y
```

</details>





