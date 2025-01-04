from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l2', C=1.0)  # C is the inverse of λ
model.fit(X_train, y_train)



this is default in logistic