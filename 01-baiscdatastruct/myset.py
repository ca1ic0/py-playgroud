#set 是一系列key的集合，hashbased
hs   = {1,2,3,4}
hs1  = {'a','b','c'}

# ADD (not addable)

# delete
# hs.pop('1') err only hs.pop()
hs.pop()
hs.remove(2)

print(hs)

# update
hs.add(1235)
print(hs)

# find

for i in hs:
    print(i)

#excepiton
try :
    hs.remove(2)
except KeyError:
    print("元素不存在")

