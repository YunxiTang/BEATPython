from functools import wraps

# usage of (*args, **kwargs)
def test_func(x, y, *args, **kwargs):
    print(x)
    print(y)
    print(args)
    print(kwargs.pop('res', None))
    print(kwargs.get('ele', None))
    print(kwargs)

    
input_1 = [90, 43, 56]
input_2 = {'res': 23, 'ele': 70}
test_func(12, 23, *input_1, **input_2)

# decorator function
def printDecorator(func):
    
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        print(f'wrapped function {func.__name__}')
        return func(*args, **kwargs)
    return wrapped_func

@printDecorator
def say_hi(name: str):
    print('Hi ' + name)
    return 'Done'
    
def get_hi_func():
    def hi():
        print('hi')
    return hi

hi_func = get_hi_func()
print(hi_func, hi_func())
print('==========')
print(say_hi('Jax'))
print(say_hi.__name__)

# decorator class
class logIt(object):
    def __init__(self, logfile: str = '/home') -> None:
        self._logfile = logfile
        
    def __call__(self, func):
        @wraps(func)
        
        def wrapped_func(*args, **kwargs):
            log_string = func.__name__ + self._logfile
            print('logging into: ' + log_string)
            self.notify()
            return func(*args, **kwargs)
        
        return wrapped_func
    
    def notify(self):
        pass
    
class EmailLogIt(logIt):
    def __init__(self, email='admin@myproject.com', *args, **kwargs) -> None:
        super(EmailLogIt, self).__init__(*args, **kwargs)
        self._email = email
        
    def notify(self):
        print('send email to ' + self._email)
        return super().notify()
    
@logIt()
def fun_o(x, y):
    return x + y

@EmailLogIt('todo@todo.com', '/data')
def fun_x(x, y):
    return x + y

print('==========')
fun_o(1, 2)
fun_x(1, 2)


def repeat(times):
    ''' call a function a number of times '''
    def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = fn(*args, **kwargs)
            return result
        return wrapper
    return decorate


@repeat(10)
def say(message):
    ''' print the message 
    Arguments
        message: the message to show
    '''
    print(message)


say('Hello')

