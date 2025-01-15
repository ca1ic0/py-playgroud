class Master(object):
    def __init__(self):
        print("Master cons")
        self.kongfu = '[古法煎饼果⼦配⽅]'

    def make_cake(self):
        print(f'运⽤{self.kongfu}制作煎饼果⼦')
    def master_method(self):
        print("master private")
 
# 创建学校类
class School(object):
    def __init__(self):
        print("School Cons")
        self.kongfu = '[⿊⻢煎饼果⼦配⽅]'

    def make_cake(self):
        print(f'运⽤{self.kongfu}制作煎饼果⼦')

class Prentice(School, Master):
    def __init__(self):
        self.kongfu = '[自创煎饼果⼦配⽅]'

    def make_cake(self):
        self.__init__()
        print(f'运⽤{self.kongfu}制作煎饼果⼦')  

    def master_make_cake(self):
        Master.__init__(self)
        Master.make_cake(self)

daqiu = Prentice()

daqiu.master_make_cake()
daqiu.make_cake()
