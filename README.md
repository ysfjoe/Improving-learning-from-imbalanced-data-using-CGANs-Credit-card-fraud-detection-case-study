# Improving-learning-from-imbalanced-data-using-CGANs-Credit-card-fraud-detection-case-study
Fraud detection concerns many financial institutions and banks as this crime costs them around $ 67 billion per year, the most common fraud is the credit card one. It is defined in Wikipedia as fraud committed using a payment card, such as a credit card or debit card with the purpose of obtaining goods or services, or to make payment to another account which is controlled by a criminal. This is a very relevant problem that demands the attention of communities such as machine learning and data science where the solution to this problem can be automated. This problem is particularly challenging from the perspective of learning, as it is characterized by various factors such as class imbalance. The number of valid transactions far outnumber fraudulent ones.
In this paper, two main parts will be discussed, the imbalanced
data problem and the approaches used, such as:
• Undersampling : RANDOM UNDER SAMPLER
• Oversampling : SMOTE and CGANS
The different models that we used in our predictions are:
• Logistic regression
• Random forest
• Xgboost
• Artificial Neural Network
• LSTM
• CNN-1D
The metric that we focused on during our predictions is the
Area Under Precision Recall Curve “AUPRC”, because it is a useful
performance metric for imbalanced data in a problem setting where
you care a lot about finding the positive examples.

we focused on the Oversampling using Conditional Generative Adversarial Networks (CGANs) which are an extension of the of the GANs model. GANs were introduced in 2014 by Ian J. Goodfellow in the article Generative Adversarial Network. Generative Adversarial Networks belong to the set of generative models. It means that they can produce / to generate new content. The difference between the GANs and CGANs is that in CGANs both the Generator and Discriminator both receive some additional conditioning input information.

In our case we found that RANDOM FOREST algorithm with CGANS for sampling is the best model.

We Developed a web application, using FLASK for back-end and Bootstrap/MaterlizeCss for the front-end, to predict whether a transaction is fraud or not, by a single transaction or by a file of transactions. we provide also in this user interface some statistics about the dataset using PLOTLY library.For collaborative work, we used MLFLOW tool to manage the MLlifecycle, including experimentation, reproducibility, and a centralmodel registry.

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. model.py - This contains code fot our Machine Learning model to predict 
2. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.
3. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
4. templates - This folder contains the HTML template to allow user to enter employee detail and displays the predicted employee salary.

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

You should be able to view the homepage as below :
![alt text](http://www.thepythonblog.com/wp-content/uploads/2019/02/Homepage.png)

Enter valid numerical values in all 3 input boxes and hit Predict.

If everything goes well, you should  be able to see the predcited salary vaule on the HTML page!
![alt text](http://www.thepythonblog.com/wp-content/uploads/2019/02/Result.png)

4. You can also send direct POST requests to FLask API using Python's inbuilt request module
Run the beow command to send the request with some pre-popuated values -
```
python request.py
```
