''' maximum likelihood '''
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import math
import seaborn as sns

sns.set_theme()
# =================== Gaussian Distribution Approximation ================== #
class Theta(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._theta = nn.Parameter(torch.tensor([-10.0, 100.0]))

    def forward(self, x):
        return 1. / (self._theta[1] * math.sqrt(2. * math.pi)) * torch.exp(-1./2 * ((x - self._theta[0]) / self._theta[1]) ** 2)
    
    def sample(self, n):
        '''
            sample data from the approximated distribution
        '''
        dist_tmp = torch.distributions.Normal(loc=self._theta.data[0], scale=self._theta.data[1])
        samples = dist_tmp.sample([n, 1])
        return samples

loc = 5.0
N = 1000
batch_size = 500
dist_q = Theta()
dist_p = torch.distributions.Normal(loc=loc, scale=1.0)
sampled_data = torch.concat((dist_p.sample([N, 1]), torch.distributions.Normal(loc=5.0, scale=1.0).sample([N, 1])), dim=0)

# set optimizer
optimizer = torch.optim.AdamW(dist_q.parameters(), 1e-3)
loss_hist = []
k = 0
fig, ax = plt.subplots(2, 5)
for epoch in range(100):
    if epoch % 10 == 0:
        tmp_samples = dist_q.sample(2*N)
        ax[k // 5, k % 5].set_xlim(-10, 10)
        ax[k // 5, k % 5].set_title(f'Epoch {epoch}')
        ax[k // 5, k % 5].hist(tmp_samples.flatten().data, bins=40, density=True)
        ax[k // 5, k % 5].hist(sampled_data.flatten(), bins=20, density=True)
        k += 1
    
    tmp = 0.
    for i in range(2*N-batch_size):
        x = sampled_data[i:i+batch_size]
        prob_q = dist_q(x)
        
        loss = torch.mean(-torch.log(prob_q))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tmp += loss.data
    print(epoch, '||', tmp)
    loss_hist.append(tmp)
plt.show()

print(dist_q._theta, loc)
fig, ax = plt.subplots(1, 2)
ax[0].set_title("nll")
ax[0].plot(loss_hist)

ax[1].set_title("samples")
ax[1].hist(sampled_data.flatten(), bins=20, density=False)
plt.show()

# ========================= binary distribution ===================
# dummy data
n_H = 75309
n_T = 74853

# Initialize our paramteres
theta = torch.tensor(0.1, requires_grad=True)

# gradient descent
lr = 1e-8
Loss = []
for iter in range(1000):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()
    Loss.append(loss.data)


print(theta, n_H / (n_H + n_T))

plt.figure()
plt.plot(Loss)
plt.show()