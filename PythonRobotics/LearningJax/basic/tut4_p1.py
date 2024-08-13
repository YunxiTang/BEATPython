import jax
import jax.numpy as jnp
from flax import linen as nn

class SomeModule(nn.Module):
    feature_dim: int
    
    def setup(self):
        # 定义SomeModule的网络结构
        self.dense_layer = nn.Dense(self.feature_dim)

    def __call__(self, inputs):
        # 调用SomeModule的网络
        return self.dense_layer(inputs)

class AnotherModule(nn.Module):
    # 将SomeModule作为参数传入
    some_module: SomeModule
    
    def setup(self):
        # 可以在这里执行一些初始化操作，如果需要的话
        self.fc1 = nn.Dense(1)
    
    def __call__(self, inputs):
        # 使用传入的SomeModule
        x = self.some_module(inputs)
        x = self.fc1(x)
        return x

# 创建SomeModule的实例
some_mod = SomeModule(feature_dim=5)

# 创建AnotherModule的实例，并传入some_mod作为参数
another_mod = AnotherModule(some_module=some_mod)

# 初始化模型参数
x = jnp.ones((10, 10))  # 假设输入数据维度为(batch_size, features)
params = another_mod.init(jax.random.PRNGKey(0), x)

# params 现在包含了another_mod的参数，包括嵌套的some_mod的参数
print(params)