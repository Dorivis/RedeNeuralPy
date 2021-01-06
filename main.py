from RedeNeural import RedeNeural
import random

rn = RedeNeural(2,3,1)

dados = {"inputs":[[1,1],[1,0],[0,1],[0,0]],"output":[[0],[1],[1],[0]]}

train = True

while (train):
    for i in range(10000):
        index = random.randint(0,3)
        rn.train(dados["inputs"][index],dados["output"][index])
    if ((rn.predict([0,0])[0]) < 0.04 and (rn.predict([1,0])[0]) > 0.98):
        train = False
        print(rn.predict([1,1]))
        print("terminou")