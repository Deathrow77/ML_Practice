import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)

num_obs = 5000

x1 = np.random.multivariate_normal([0,0], [[1, .75], [.75,1]], num_obs)
x2 = np.random.multivariate_normal([1,4], [[1, .75], [.75,1]], num_obs)


labels = np.hstack((np.zeros(num_obs), np.ones(num_obs)))
features = np.vstack((x1,x2)).astype(np.float32)
plt.figure(figsize=(12,8))
plt.scatter(features[:, 0], features[:, 1], c=labels, alpha=0.4)
plt.show()




# Logistic Regression Model


def sigmoid(scores):
    return (1/(1+np.exp(-scores)))


def log_likelihood(features, weights, target):
    scores = np.dot(features, weights)
    ll = np.sum(target*scores + np.log(1+np.exp(scores)))
    return ll

def Logistic_Regression(features, target, num_steps, learning_rate, add_intercepts=False):
    if add_intercepts:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    for step in range(num_obs):
        scores = np.dot(features, weights)
        prediction = sigmoid(scores)

        # Optimizing and updating weights
        output_error_signal = target - prediction
        
        gradient = np.dot(features.T, output_error_signal)
        weights = learning_rate*gradient

        if step%10000==0:
            print(log_likelihood(features, weights, target))
        
    return weights

weights = Logistic_Regression(features, labels, num_steps=30000, learning_rate=5e-5, add_intercepts=True)
print(weights)