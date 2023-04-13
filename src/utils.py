class InvertedResidual(nn.Module):
  def __init__(self,input, output, stride, expand_ratio):
    super().__init__()
    self.stride = stride

    assert stride in [1,2]
     
    hidden_dim = int(round(input*expand_ratio))
    self.use_res_conn = (input == output and self.stride == 1)

    layers = []
    if expand_ratio!=1:
      layers.extend([
          nn.Conv2d(input,hidden_dim,1,1,0,bias=False),
          nn.BatchNorm2d(hidden_dim),
          nn.ReLU6(inplace=True)
      ])

    layers.extend(
        [ 
          nn.Conv2d(hidden_dim,hidden_dim,3,self.stride,1,groups=hidden_dim,bias=False),
          nn.BatchNorm2d(hidden_dim),
          nn.ReLU6(inplace=True),

          nn.Conv2d(hidden_dim,output,1, 1, 0, bias=False),
          nn.BatchNorm2d(output)
        ]
    )

    self.conv = nn.Sequential(*layers)
    self.output_channels = output
    self._is_cn = self.stride>1   
  
  def forward(self,x):
    if self.use_res_conn:
      return x+self.conv(x)
    else:
      return self.conv(x)
    
"""
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
"""
def make_divisible(value, divisor):
  new_value = max(divisor, int(value + divisor/2)//divisor * divisor)
  if new_value < 0.9*value:
    new_value += divisor
  return int(new_value)

