# Collection of some advanced skills while using python

## decoeator
A function of function. Modify the behavior of a function, e.g. add new properties, without changing the function itself. See script [test_decorator.py](./test_decorator.py)

## classmethod 
The function decorated by `@classmethod` does not need to be instantiated and does not need a `self` parameter. 

The first parameter needs to be the `cls` parameter representing its own class, which can be used to call class attributes, class methods, instantiated objects, etc. See script [use_classmethod.py](./use_classmethod.py)

`cls`: represents the outer class itself, which can be instantiated or directly call static methods, class methods, and static variables. 
* To call a classmethod, the general form is: `class_name.classmethod_name()`. For example,
```
class Link:
    '''
        Robot link
    '''
    def __init__(self, length: float, mass: float):
        self.mass = mass
        self.length = length

    @classmethod
    def copy_from_link(cls, link):
        return cls(link.length, link.mass)
    

if __name__ == '__main__':
    link1 = Link(0.5, 1.0)
    link2 = Link.copy_from_link(link1)
    link3 = link1.copy_from_link(link1) # not recommend
    
```
Also, we can instantiate a class object and call the class method using this object, which is not recommended.

* To call static variable
Two ways:
```
# way one
class_name.variable_name
# way two
cls.variable_name
```

* To call other class methods/static methods inside classmethods
```
cls.classmethod_name()
cls.staticmethod_name()
```

* To call common methods inside classmethods to access instance properties. 
You need to instantiate a class object through the `cls` variable, and then use this object to call common methods and the instance's properties.
```
cls().method_name()
```

## dataclass


## MultiProcess
* See script [multi_process.py](multi_process.py) for a simple sample code;
* For communications with `Pipe` between two processes, see [multi_process_com.py](multi_process_com.py);
* For communications with `Queue` between multiple processes, see [multi_process_com2.py](Advanced/multi_process_com2.py)