import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import pickle

# stats for prediction
pWGPA = 4.1
pSAT = 1330
pACT = 28

pWGPA2 = 4.17
pSAT2 = 1330
pACT2 = 31

pWGPA3 = 3.14
pSAT3 = 1030
pACT3 = 43

# are we still training?
training = False

# reading in .csv with pandas
data = pd.read_csv("UMDdata.csv")
testdata = pd.read_csv("UMDtest.csv")

print(data)

# set up train_test_split data
WGPA = list(data["WGPA"])
SAT = list(data["SAT"])
ACT = list(data["ACT"])
result = list(data["result"])

X = list(zip(WGPA, SAT, ACT))
y = list(result)

# set up testing data
test_WGPA = list(testdata["WGPA"])
test_SAT = list(testdata["SAT"])
test_ACT = list(testdata["ACT"])
test_result = list(testdata["result"])

test_x = list(zip(test_WGPA, test_SAT, test_ACT))
test_y = list(test_result)

# train model 'n' amount of times and save the highest accuracy
n = 3000
best = 0
if training:
    for _ in range(n):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

        model = KNeighborsClassifier(n_neighbors=13)
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)

        # if the accuracy of the current model is higher than the saved model, replace the saved model with the current
        if acc > best:
            best = acc
            print(best)
            knnPickle = open('UMDModel.pickel', 'wb')
            pickle.dump(model, knnPickle)

# loads 'model' to be pickel file and tests against test data
model = pickle.load(open("UMDModel.pickel", "rb"))
print("Saved model against test data = ", model.score(test_x, test_y))

# test model against test data
predicted = model.predict(test_x)

for x in range(len(predicted)):
    print("Actual: ", test_y[x], "Data: ", test_x[x], "Predicted: ", predicted[x])


# predicts the chances with pWGPA, pSAT, pACT
prediction = [[[pWGPA, pSAT, pACT]],
              [[pWGPA2, pSAT2, pACT2]],
              [[pWGPA3, pSAT3, pACT3]]]
y = 0
for x in prediction:
    if model.predict(x) == 1:
        print("WGPA:", prediction[y][0][0], "SAT:", prediction[y][0][1], "ACT:", prediction[y][0][2], "----> You got in!")
        y += 1
    else:
        print("WGPA:", prediction[y][0][0], "SAT:", prediction[y][0][1], "ACT:", prediction[y][0][2], "----> Sorry, denied.")
        y += 1
