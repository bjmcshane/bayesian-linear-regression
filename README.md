# bayesian-linear-regression
Bayesian and regularized linear regression. This was an assignment for my graduate level machine learning course.

## Results
### Part 1: Regularization
These are plots of training set MSE and test set MS as a function of the regularization parameter lambda

![alt text](https://github.com/bjmcshane/bayesian-linear-regression/blob/main/images/crime.png?raw=true)
![alt text](https://github.com/bjmcshane/bayesian-linear-regression/blob/main/images/wine.png?raw=true)
![alt text](https://github.com/bjmcshane/bayesian-linear-regression/blob/main/images/artsmall.png?raw=true)
![alt text](https://github.com/bjmcshane/bayesian-linear-regression/blob/main/images/artlarge.png?raw=true)


### Part 2: Model Selection using Cross Validation
I used 10 fold cross validation on the training set to pick a value for lambda in the 0-150 range, and then retrained on the entire training set using that lambda and evaluated on the testing set. These are the results.


![alt text](https://github.com/bjmcshane/bayesian-linear-regression/blob/main/images/part2.png?raw=true)

### Part 3: Bayesian Model Selection
In this part I had to iteratively apply an update function to the parameters of a bayesian linear regression model. After the model is tuned and tested, these are the results.

![alt text](https://github.com/bjmcshane/bayesian-linear-regression/blob/main/images/part3.png?raw=true)
