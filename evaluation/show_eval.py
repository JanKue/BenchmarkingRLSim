from numpy import load

data = load('reach/evaluations.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])