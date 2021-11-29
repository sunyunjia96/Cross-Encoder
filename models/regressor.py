import torch
import torch.nn as nn

class regressor(nn.Module):
    def __init__(self, rdim, n_out=3):
        super(regressor,self).__init__()
       
        self.before = nn.Sequential(
        nn.BatchNorm1d(rdim),
        nn.Tanh(),
        )

        self.regressor = nn.Sequential(
        #Add the dimension of head pose
        nn.Linear(rdim+3,rdim),
        nn.Tanh(),
        )

        self.fc = nn.Linear(rdim,n_out)

    def forward(self,x,head):
        x = torch.cat((x,head),dim=1)
        x = self.regressor(x)
        x = self.fc(x)
        return x
