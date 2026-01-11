
class RCB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCB, self).__init__()

        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)
        )

        self.act = act


    def forward(self, x):
        res = self.body(x)
        res = self.act(self.gcnet(res))
        res += x
        return res
