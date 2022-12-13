from numpy import load

data = load('open_door/sac/evaluations.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])
