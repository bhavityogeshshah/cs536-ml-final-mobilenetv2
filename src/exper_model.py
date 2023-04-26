from utils import InvertedResidual, Conv1x1BN, ConvBN, make_divisible
from torch import nn
import torch

class MobileNetV2(nn.Module):
  def __init__(
      self, 
      num_classes=1000,
      width = 1,
      input_size=32, # changed from 224 to 32
      dropout=0.2
      ):
    super().__init__()

    block = InvertedResidual
    input_channel = 32
    self.last_channel = 1280

    inverted_residual_setting = [
        [1, 16, 1, 1],
        # [6, 24, 2, 2], removing one inverted residual block
        [6, 32, 3, 1], # changed the stride from 2 to 1
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    assert input_size%32==0

    round_nearest = 8
    input_channel = make_divisible(input_channel*width, round_nearest)
    self.last_channel = make_divisible(self.last_channel*max(width,1), round_nearest)

    features = [
      ConvBN(3,input_channel,1)  # changed the stride from 2 to 1
    ]

    for t, c, n, s in inverted_residual_setting:
      output_channel = make_divisible(c*width, round_nearest)
      
      for i in range(n):
        stride = s if i==0 else 1
        features.append(block(input_channel,output_channel,stride, t))
        input_channel = output_channel

    features.append(Conv1x1BN(input_channel,self.last_channel))

    self.features = nn.Sequential(*features)
    self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )
    
    self._init_weights()
  
  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight, mode="fan_out")
          if m.bias is not None:
              nn.init.zeros_(m.bias)
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
          nn.init.ones_(m.weight)
          nn.init.zeros_(m.bias)
      elif isinstance(m, nn.Linear):
          nn.init.normal_(m.weight, 0, 0.01)
          nn.init.zeros_(m.bias)

  def _forward(self,x):
    x = self.features(x)
    x = nn.functional.adaptive_avg_pool2d(x,(1,1))
    x = torch.flatten(x,1)
    x = self.classifier(x)
    return x

  def forward(self,x):
    return self._forward(x)