import pickle

with open('metrics.txt', 'rb') as handle:

    myvar = pickle.load(handle)

print(myvar)
