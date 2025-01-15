import os 

# def recursive_find(filename , dir):
#     for x in os.listdir(dir):
#         full_path = os.path.join(dir, x)
#         if os.path.isdir(x):
#             recursive_find(filename, full_path)
#         if os.path.isfile(x):
#             if filename in x:
#                 print(x)

def recursive_find(filename, directory):
    for x in os.listdir(directory):
        full_path = os.path.join(directory, x)
        if os.path.isdir(full_path):
            recursive_find(filename, full_path)
        elif os.path.isfile(full_path):
            if filename in x:
                print(full_path)


recursive_find("test.csv","/home/calico/code/py-playgroud/")