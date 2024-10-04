def disp(a, b, *args, c, d, **kwargs):
    print(a, b)
    for arg in args:
        print(arg)
    print(c, d)
    
    for key, val in kwargs.items():
        print(key, val)
        

if __name__ == '__main__':
    disp(12, 13, 45, 46, c='x', d='y', q='point')