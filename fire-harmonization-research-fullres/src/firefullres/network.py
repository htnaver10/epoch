"""Compact trainable native-band CNN with a coordinate-aware VIIRS-query head."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        groups=min(8,cout)
        while cout%groups:groups-=1
        self.layers=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(groups,cout),nn.SiLU(),
                                  nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(groups,cout),nn.SiLU())

    def forward(self,x):return self.layers(x)


class NativeFireNet(nn.Module):
    def __init__(self, base=32, branch=8, activation_checkpointing=False):
        super().__init__()
        self.activation_checkpointing=activation_checkpointing
        self.thermal=Block(12,base);self.ref500=Block(10,branch);self.ref250=Block(4,branch);self.geo=Block(14,branch)
        self.fuse=Block(base+3*branch,base)
        self.down1=Block(base,2*base);self.down2=Block(2*base,4*base);self.down3=Block(4*base,8*base)
        self.up2=Block(12*base,4*base);self.up1=Block(6*base,2*base);self.up0=Block(3*base,base)
        self.native_head=nn.Conv2d(base,1,1)
        # No normalization or spatial mixing over padded query canvases.
        self.query_head=nn.Sequential(nn.Conv2d(base+3*branch,base,1),nn.SiLU(),nn.Conv2d(base,1,1))

    def block(self,layer,x):
        return checkpoint(layer,x,use_reentrant=False) if self.activation_checkpointing and self.training else layer(x)

    @staticmethod
    def resize(x,shape):return F.interpolate(x,size=shape,mode='bilinear',align_corners=False)

    @staticmethod
    def sample(x,grid):
        with torch.autocast(device_type=x.device.type,enabled=False):
            return F.grid_sample(x.float(),grid.float(),mode='bilinear',padding_mode='zeros',align_corners=False)

    def forward(self,thermal,reflective_500,reflective_250,geometry,query_grid):
        a=self.block(self.thermal,thermal);b=self.block(self.ref500,reflective_500)
        c=self.block(self.ref250,reflective_250);g=self.block(self.geo,geometry)
        x0=self.block(self.fuse,torch.cat([a,self.resize(b,a.shape[-2:]),self.resize(c,a.shape[-2:]),g],1))
        x1=self.block(self.down1,F.avg_pool2d(x0,2));x2=self.block(self.down2,F.avg_pool2d(x1,2))
        x3=self.block(self.down3,F.avg_pool2d(x2,2))
        y2=self.block(self.up2,torch.cat([self.resize(x3,x2.shape[-2:]),x2],1))
        y1=self.block(self.up1,torch.cat([self.resize(y2,x1.shape[-2:]),x1],1))
        y0=self.block(self.up0,torch.cat([self.resize(y1,x0.shape[-2:]),x0],1))
        fine=torch.cat([self.sample(f,query_grid) for f in [y0,b,c,g]],1)
        return {'logits':self.query_head(fine),'native_logits':self.native_head(y0)}
