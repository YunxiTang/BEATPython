# Mid Part
## Mixin
Mixin 常被译为“混入”，是一种编程模式，在 Python 等面向对象语言中，通常它是实现了某种功能单元的类，用于被其他子类继承，将功能组合到子类中.
利用 Python 的多重继承，子类可以继承不同功能的 Mixin 类，按需动态组合使用。
当多个类都实现了同一种功能时，这时应该考虑将该功能抽离成 Mixin 类.
Note: the order of base classs is: from right to left!

### Properties
#### Mixin 实现的功能需要是通用的，并且是单一的, Mixin 类都适用于大部分子类，每个 Mixin 只实现一种功能，可按需继承.
#### Mixin 只用于拓展子类的功能，不能影响子类的主要功能，子类也不能依赖 Mixin.
#### 继承不同的 Mixin 只是增加一些功能，并不影响自身的主要功能。如果是依赖关系，则是真正的基类，不应该用 Mixin 命名.
#### Mixin 类自身不能进行实例化，仅用于被子类继承.