import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Load data
msg = pd.read_csv('document.csv', names=['message', 'label'])
print("Total Instances of Dataset: ", msg.shape[0])

# Map labels to numerical values
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

# Split data into training and testing sets
X = msg.message
y = msg.labelnum
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=42)

# Vectorize text data
count_v = CountVectorizer()
Xtrain_dm = count_v.fit_transform(Xtrain)
Xtest_dm = count_v.transform(Xtest)

# Create DataFrame to inspect features (optional)
df = pd.DataFrame(Xtrain_dm.toarray(), columns=count_v.get_feature_names_out())
print(df.head())

# Train the classifier
clf = MultinomialNB()
clf.fit(Xtrain_dm, ytrain)

# Make predictions
pred = clf.predict(Xtest_dm)

# Display predictions (optional)
for doc, p in zip(Xtest, pred):
    p = 'pos' if p == 1 else 'neg'
    print(f"{doc} -> {p}")

# Evaluate the classifier
print('Accuracy Metrics:\n')
print('Accuracy: ', accuracy_score(ytest, pred))
print('Recall: ', recall_score(ytest, pred))
print('Precision: ', precision_score(ytest, pred))
print('Confusion Matrix:\n', confusion_matrix(ytest, pred))
