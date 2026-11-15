"""Teacher consumes only VIIRS radiance/validity; student remains NativeFireNet."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from ..network import Block


class VIIRSTeacher(nn.Module):
    def __init__(self,base=16):
        super().__init__()
        self.enc0=Block(4,base);self.enc1=Block(base,base*2);self.enc2=Block(base*2,base*4)
        self.middle=Block(base*4,base*8)
        self.dec2=Block(base*12,base*4);self.dec1=Block(base*6,base*2);self.dec0=Block(base*3,base)
        self.head=nn.Conv2d(base,1,1)

    def initialize_prior(self,p):
        if not math.isfinite(p) or not 0<p<1:raise ValueError('Invalid training-only teacher prior')
        with torch.no_grad():
            self.head.weight.zero_();self.head.bias.fill_(math.log(p/(1-p)))

    def forward(self,image):
        if image.ndim!=4 or image.shape[1]!=4 or image.shape[0]!=1:
            raise ValueError('Teacher processes one unpadded four-channel VIIRS crop at a time')
        if min(image.shape[-2:])<8:raise ValueError('Teacher crop is too small')
        a=self.enc0(image);b=self.enc1(F.avg_pool2d(a,2));c=self.enc2(F.avg_pool2d(b,2))
        x=self.middle(F.avg_pool2d(c,2))
        for skip,block in ((c,self.dec2),(b,self.dec1),(a,self.dec0)):
            x=block(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
        return self.head(x)


def frozen_teacher(model):
    model.eval();model.requires_grad_(False)
    return model


def bernoulli_kl_sum(student_logits,teacher_logits,weight,temperature):
    """T^2 * sum w KL(Ber(sigmoid(teacher/T)) || Ber(sigmoid(student/T)))."""
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('Distillation temperature must be positive')
    if student_logits.shape!=teacher_logits.shape or student_logits.shape!=weight.shape:
        raise ValueError('Distillation logits and weights must have identical shapes')
    if not torch.isfinite(teacher_logits).all() or not torch.isfinite(weight).all() or (weight<0).any():
        raise ValueError('Distillation targets must be finite with nonnegative weights')
    teacher=teacher_logits.detach().float()/temperature
    student=student_logits.float()/temperature
    # Both Bernoulli classes are essential; a one-channel softmax has zero loss.
    positive=F.kl_div(F.logsigmoid(student),F.logsigmoid(teacher),reduction='none',log_target=True)
    negative=F.kl_div(F.logsigmoid(-student),F.logsigmoid(-teacher),reduction='none',log_target=True)
    return ((positive+negative)*weight).sum()*temperature**2
