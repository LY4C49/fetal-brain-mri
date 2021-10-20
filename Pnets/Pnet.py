from nets.Pnet_parts import block1,block2,block3,block4,block5,block6
from torch import nn as nn

class PN(nn.Module):
    def __init__(self):
        super(PN,self).__init__()
        self.b1=block1()
        self.b2=block2()
        self.b3=block3()
        self.b4=block4()
        self.b5=block5()
        self.b6=block6()

    def forward(self,x):
        r1=self.b1(x)
        r2=self.b2(r1)
        r3=self.b3(r2)
        r4=self.b4(r3)
        r5=self.b5(r4)
        r6=self.b6(r1,r2,r3,r4,r5)
        return r6

