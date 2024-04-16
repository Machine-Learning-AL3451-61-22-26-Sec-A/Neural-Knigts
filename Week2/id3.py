import streamlit as st
import pandas as pd
import math
import copy

attribute = ['outlook', 'temp', 'humidity', 'wind']

class Node(object):
    def __init__(self):
        self.value = None
        self.decision = None
        self.childs = None

def findEntropy(data, rows):
    yes = 0
    no = 0
    ans = -1
    idx = len(data[0]) - 1
    entropy = 0
    for i in rows:
        if data[i][idx] == 'Yes':
            yes += 1
        else:
            no += 1

    x = yes/(yes+no) if (yes+no) != 0 else 0
    y = no/(yes+no) if (yes+no) != 0 else 0
    if x != 0 and y != 0:
        entropy = -1 * (x*math.log2(x) + y*math.log2(y))
    if x == 1:
        ans = 1
    if y == 1:
        ans = 0
    return entropy, ans

def findMaxGain(data, rows, columns):
    maxGain = 0
    retidx = -1
    entropy, ans = findEntropy(data, rows)
    if entropy == 0:
        return maxGain, retidx, ans

    for j in columns:
        mydict = {}
        idx = j
        for i in rows:
            key = data[i][idx]
            if key not in mydict:
                mydict[key] = 1
            else:
                mydict[key] += 1
        gain = entropy

        for key in mydict:
            yes = 0
            no = 0
            for k in rows:
                if data[k][j] == key:
                    if data[k][-1] == 'Yes':
                        yes += 1
                    else:
                        no += 1
            x = yes/(yes+no) if (yes+no) != 0 else 0
            y = no/(yes+no) if (yes+no) != 0 else 0
            if x != 0 and y != 0:
                gain += (mydict[key] * (x*math.log2(x) + y*math.log2(y)))/(yes+no)
        if gain > maxGain:
            maxGain = gain
            retidx = j
    return maxGain, retidx, ans

def buildTree(data, rows, columns):
    maxGain, idx, ans = findMaxGain(data, rows, columns)
    root = Node()
    root.childs = []
    if maxGain == 0:
        if ans == 1:
            root.value = 'Yes'
        else:
            root.value = 'No'
        return root

    root.value = attribute[idx]
    mydict = {}
    for i in rows:
        key = data[i][idx]
        if key not in mydict:
            mydict[key] = 1
        else:
            mydict[key] += 1

    newcolumns = copy.deepcopy(columns)
    newcolumns.remove(idx)
    for key in mydict:
        newrows = []
        for i in rows:
            if data[i][idx] == key:
                newrows.append(i)
        temp = buildTree(data, newrows, newcolumns)
        temp.decision = key
        root.childs.append(temp)
    return root

def main():
    st.title("Decision Tree Classifier")

    # Upload CSV file
    uploaded_file = st.file_uploader("trainingdata.csv", type=['csv'])
    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)

        # Build decision tree when button is clicked
        if st.button("Build Decision Tree"):
            X = dataset.values
            rows = [i for i in range(len(X))]
            columns = [i for i in range(len(X[0]))]
            root = buildTree(X, rows, columns)  # Pass X as an argument
            root.decision = 'Start'

            # Display decision tree
            st.write("Decision Tree:")
            # Printing decision tree
            print_tree(root)

# Function to print decision tree
def print_tree(node, depth=0):
    if node is None:
        return
    if node.decision is not None:
        st.write("  " * depth + f"- {node.decision}: {node.value}")
    for child in node.childs:
        print_tree(child, depth + 1)

# Run the app
if __name__ == "__main__":
    main()
