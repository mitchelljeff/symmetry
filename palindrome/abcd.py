from random import random

train=100000
test =10000
dev  =10000

with open("abcdtrain","w") as trainfile, open("abcdtest","w") as testfile, open("abcddev","w") as devfile:
    for l in range(7,13):
        for i in range(4**l):
            r=random()*(4**l)*6
            if r<(train+test+dev):
                seq=list()
                binary=("{0:0"+str(2*l)+"b}").format(i)
                pairs=[binary[i:i+2] for i in range(0,len(binary),2)]
                for char in pairs:
                    if char=="00":
                        seq.append("a")
                    if char=="01":
                        seq.append("b")
                    if char=="10":
                        seq.append("c")
                    if char=="11":
                        seq.append("d")
                seq.append("o")
                pairs.reverse()
                for char in pairs:
                    if char=="00":
                        seq.append("a")
                    if char=="01":
                        seq.append("b")
                    if char=="10":
                        seq.append("c")
                    if char=="11":
                        seq.append("d")
                if r < test:
                    testfile.write(" ".join(seq)+"\n")
                elif r < dev+test:
                    devfile.write(" ".join(seq)+"\n")
                elif r < train+test+dev:
                    trainfile.write(" ".join(seq)+"\n")

