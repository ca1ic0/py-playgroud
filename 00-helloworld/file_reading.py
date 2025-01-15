import os

with open("./test.csv" ,"r") as f :
    for i,line in enumerate(f.readlines()):
        print(i,line)

with open("./test.csv" ,"r") as f :
    for line in f.readlines():
        print(line)

with open("./test.csv" ,"r") as f :
    for total in f.read(): # 单个字符形式的读取
        print(total)

with open("./test.csv" ,"r") as f : # 单个字符形式的读取
    print(f.read())