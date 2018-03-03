import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
#for i in range(len(iris.target)):
#    print("Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))
test_index = [0, 50, 100]

#training data
train_target = np.delete(iris.target, test_index)
train_data = np.delete(iris.data, test_index, axis = 0)

#testing data
test_target = iris.target[test_index]
test_data = iris.data[test_index]

classifier = tree.DecisionTreeClassifier()
classifier.fit(train_data, train_target)

print("Expected: ", test_target)
print("Actual: ", classifier.predict(test_data))

from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(classifier,
    out_file=dot_data,
    feature_names = iris.feature_names,
    class_names = iris.target_names,
    filled=True, rounded = True,
    impurity = False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
