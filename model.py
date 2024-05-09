import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('Student_Marks.csv')

X = data[['number_courses', 'time_study']]
y = data['Marks']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n \t\tStudent Mark Prediction Model Developed By Hamsani\n\n")
while True:
    i = input("Enter Student Name (or 'exit' to quit): ")
    if i == "exit":
        print("Thank You!!! Have a Nice Day")
        break
    else:
        nc = (input(f"Enter the number of courses enrolled by {i}: "))
        t = (input(f"Enter the study time of {i}: "))
        new_data = pd.DataFrame({'number_courses': [nc], 'time_study': [t]})
        new_predictions = model.predict(new_data)
        if(new_predictions >= 100):
            new_predictions = 100;
        print(f"The Average Predicted marks for {i} is: {new_predictions}\n")
    


