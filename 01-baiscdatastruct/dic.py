#create 
dic = {'a' : 100 , 'b':200 , 'c' : 300}

# k-v k is not changeable 

# ADD
dic['abc'] = 102 

# UPDATE
dic['a'] = 120

# DELETE
dic.pop('a')

# FIND
print(dic['abc'])
print(dic.get('abc'))

# foreach key
for k in dic:
    print(k)

# elem
for k,v in dic.items():
    print(k,v)

#exception keyerror
try:
    a = dic['a']
except KeyError:
    print("元素不存在")

if dic.get('a') == None:
    print("元素不存在")
else:
    print(dic.get('a'))