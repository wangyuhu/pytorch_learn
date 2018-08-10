import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size())

net=Net(n_feature=1,n_hidden=10,n_output=1)

optimizer=torch.optim.SGD(net.parameters(),lr=0.3)
loss_func=torch.nn.MSELoss()

plt.ion() # something about plotting

for t in range(400):
    prediction=net(x)
    loss=loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ##show
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()    
##show
