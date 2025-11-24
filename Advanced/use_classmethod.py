class Link:
    '''
        Robot link
    '''

    # static (class) variables
    type = "regular_link"

    def __init__(self, length: float = 1.0, mass: float = 1.0):
        self.mass = mass
        self.length = length

    @staticmethod
    def print_type():
        print('Type: {}'.format(Link.type))

    @classmethod
    def copy_from_link(cls, link):
        link = cls(link.length, link.mass)
        link.add_CoM([0.3, 0.3, 0.3])
        return link
    
    def add_CoM(self, com: list[float]):
        self.CoM = com
    
    def __repr__(self) -> str:
        if hasattr(self, 'CoM'):
             rep = 'link length: {} \nlink mass: {} \nlink com: {}'.format(self.length, 
                                                                           self.mass,
                                                                           self.CoM)
        else:
            rep = 'link length: {} \nlink mass: {}'.format(self.length,
                                                                 self.mass,
                                                                           )
        return rep
    

if __name__ == '__main__':
    print('==================================')
    link1 = Link(0.5, 1.0)
    print(link1)
    print('==================================')
    link2 = Link.copy_from_link(link1)
    print(link2)
    print('==================================')
    link1.add_CoM([0.1, 0.2, 0.3])
    print(link1)
    print('==================================')
    Link.print_type()

    
    