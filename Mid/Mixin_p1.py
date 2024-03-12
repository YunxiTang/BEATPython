"""Mixin"""
class MappingMixin:
    def __getitem__(self, key):
        return self.__dict__.get(key)

    def __setitem__(self, key, value):
        return self.__dict__.set(key, value)
    

class ReprMixin:
    def __repr__(self):
        s = self.__class__.__name__ + '('
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                s += '{}={}, '.format(k, v)
        s = s.rstrip(', ') + ')'
        return s


class Person(MappingMixin, ReprMixin):
    include_keys = ('name')
    exclude_keys = tuple()
    def __init__(self, name, gender, age):
        # super.__init__()
        self.name = name
        self.gender = gender
        self.age = age




if __name__ == '__main__':
    p = Person('Jack', 'male', 25)
    print(p.__dict__)
    
    for key, val in p.__dict__.items():
        print(key, val)

    print(Person.__dict__)
    # print(p.name)
    print(p['name'])
    print(p)