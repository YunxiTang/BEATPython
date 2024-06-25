'''
    class.__dict__
'''

class A:
    a = 1
    b = 2

    def __init__(self, name:str):
        self.name = name

    def show_name(self):
        print(self.name)


if __name__ == '__main__':
    a = A('tyx')
    for key, val in A.__dict__.items():
        print(key, ':', val)