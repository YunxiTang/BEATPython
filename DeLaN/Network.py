import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
import random

torch.manual_seed(1)

class Head(nn.Module):
    """The common head for DeLaN. (ReLu Network)
       Input: q
       state_size: the dimension of q (angle position)
       output_size: the dimension of head module output
       output: For next three NNs (G(q), M(q), C(q,dq)). It is also possible to add friction term into model.
    """

    def __init__(self, state_size, output_size):
        super(Head, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(512, 512)
        self.fc10 = nn.Linear(512, 512)
        self.fc11 = nn.Linear(512, 512)
        self.fc12 = nn.Linear(512, 512)
        self.fc13 = nn.Linear(512, 512)
        self.fc14 = nn.Linear(512, 512)
        self.fc15 = nn.Linear(512, 256)
        self.fc16 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        x = F.softplus(self.fc3(x))
        x = F.softplus(self.fc4(x))
        x = F.softplus(self.fc5(x))
        x = F.softplus(self.fc6(x))
        x = F.softplus(self.fc7(x))
        x = F.softplus(self.fc8(x))
        x = F.softplus(self.fc9(x))
        x = F.softplus(self.fc10(x))
        x = F.softplus(self.fc11(x))
        x = F.softplus(self.fc12(x))
        x = F.softplus(self.fc13(x))
        x = F.softplus(self.fc14(x))
        x = F.softplus(self.fc15(x))
        x = F.softplus(self.fc16(x))
        return x


class G_net(nn.Module):
    """ Network for Gravity term G(q). (Linear network)
    """

    def __init__(self, input_size, output_size):
        super(G_net, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x


class ld_net(nn.Module):
    """Network for ld term ld(q). (Relu network) [Dialog entries for DeLaN]
    """

    def __init__(self, input_size, output_size):
        super(ld_net, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        x = F.softplus(self.fc3(x))
        x = F.softplus(self.fc4(x))
        x = F.softplus(self.fc5(x))
        x = F.softplus(self.fc6(x))
        x = F.softplus(self.fc7(x))
        x = F.softplus(self.fc8(x))
        return x


class lo_net(nn.Module):
    """Network for lo term lo(q). (Linear network). [off dialog entries for DeLaN]
    """

    def __init__(self, input_size, output_size):
        super(lo_net, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)

        return x


def act_der(nc_type, x):
    """compute the derivative of action function"""
    if nc_type == 'ReLu':
        if x <= 0:
            der = 0
        else:
            der = 1
    elif nc_type == 'Linear':
        der = 1
    return der


class DeLaN(nn.Module):
    """Network for The whole DeLaN"""

    def __init__(self, input_size, output_size_of_head):
        super(DeLaN, self).__init__()
        self.out_head = output_size_of_head
        self.in_g = int(2 * output_size_of_head / 7)
        self.in_ld = int(2 * output_size_of_head / 7)
        self.in_lo = self.out_head - self.in_g - self.in_ld

        self.Head = Head(2 * input_size, self.out_head)
        self.G_term = G_net(self.in_g, 2)
        self.ld_term = ld_net(self.in_ld, 2)
        self.lo_term = lo_net(self.in_lo, 1)

    def foward(self, x, dx, ddx):
        """input x is q, dx is d_dot, ddx is q_double dot"""
        sin_q1 = torch.sin(x[0]).unsqueeze(0)
        cos_q1 = torch.cos(x[0]).unsqueeze(0)

        sin_q2 = torch.sin(x[1]).unsqueeze(0)
        cos_q2 = torch.cos(x[1]).unsqueeze(0)

        m = torch.cat((sin_q1, cos_q1, sin_q2, cos_q2), dim=0)
        out_of_head = self.Head.forward(m)

        input_of_g = out_of_head[0:self.in_g]
        input_of_ld = out_of_head[self.in_g:self.in_g + self.in_ld]
        input_of_lo = out_of_head[(self.in_g + self.in_ld):]

        g_out = self.G_term.forward(input_of_g)
        ld_out = self.ld_term.forward(input_of_ld)
        lo_out = self.lo_term.forward(input_of_lo)

        # create a variable called l for convenience
        l = torch.cat((ld_out, lo_out), 0)

        # reshape of ld and lo
        g_out = g_out.reshape(len(g_out), )
        ld_out_2 = ld_out.diag()
        lo_out_2 = torch.tensor([[0, 0], [lo_out, 0]])

        L = ld_out_2 + lo_out_2
        L_t = torch.transpose(L, 0, 1)

        # Inertia matrix
        H = torch.matmul(L, L_t)

        # carefully compute the derivatives of dH/dt <-- dL/dt <-- dl/dt
        dl1_dq = torch.autograd.grad(outputs=l, inputs=x, grad_outputs=torch.tensor([1., 0., 0.]), retain_graph=True)[0]
        dl2_dq = torch.autograd.grad(outputs=l, inputs=x, grad_outputs=torch.tensor([0., 1., 0.]), retain_graph=True)[0]
        dl3_dq = torch.autograd.grad(outputs=l, inputs=x, grad_outputs=torch.tensor([0., 0., 1.]), retain_graph=True)[0]
        dl_dq = torch.cat((dl1_dq, dl2_dq, dl3_dq), dim=0).reshape(3, 2)

        dl_dt = torch.matmul(dl_dq, dx)
        dL_dt = torch.tensor([[dl_dt[0], 0],
                              [dl_dt[2], dl_dt[1]]])

        # compute dH/dt = L*dL'/dt + dL/dt*L'
        dH_dt = torch.matmul(L, torch.transpose(dL_dt, 0, 1)) + torch.matmul(dL_dt, L_t)

        # compute qd'Hqd --> C
        K_E = torch.matmul(torch.matmul(dx.reshape(1, 2), H), dx.reshape(2, 1)).squeeze()
        C = 1/2 * torch.autograd.grad(outputs=K_E, inputs=x, retain_graph=True)[0].reshape(2,)

        # compute the torque
        tau_out = g_out.reshape(2, ) + torch.matmul(H, ddx.reshape(2, )) + torch.matmul(dH_dt, dx.reshape(2,)) - C
        return tau_out


# test this module
if __name__ == '__main__':
    # test Head module
    x = torch.tensor([-2., 3.], requires_grad=True)
    dx = torch.tensor([1., 1.])
    ddx = torch.tensor([1., 1.])
    model = DeLaN(2, 450)
    tau = model.foward(x, dx, ddx)
    print(tau)


