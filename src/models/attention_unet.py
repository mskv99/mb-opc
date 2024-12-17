import torch
import torch.nn as nn
from torchinfo import summary

# Define an Attention Block
class AttentionBlock(nn.Module):
  def __init__(self, F_g, F_l, F_int):
    super(AttentionBlock, self).__init__()
    # Global and local attention layers
    self.W_g = nn.Sequential(
      nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(F_int)
    )

    self.W_x = nn.Sequential(
      nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(F_int)
    )

    # Attention coefficients
    self.psi = nn.Sequential(
      nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(1),
      nn.Sigmoid()
    )

    self.relu = nn.ReLU(inplace=True)

  def forward(self, g, x):
    g1 = self.W_g(g)  # Global features
    x1 = self.W_x(x)  # Local features
    psi = self.relu(g1 + x1)
    psi = self.psi(psi)  # Attention map
    return x * psi  # Weighted local features


# Define the Generator with U-Net and Attention Blocks
class AttentionUNetGenerator(nn.Module):
  def __init__(self, in_channels=1, out_channels=1):
    super(AttentionUNetGenerator, self).__init__()

    # Encoder layers
    self.encoder1 = self.conv_block(in_channels, 64)
    self.encoder2 = self.conv_block(64, 128)
    self.encoder3 = self.conv_block(128, 256)
    self.encoder4 = self.conv_block(256, 512)

    # Bottleneck
    self.bottleneck = self.conv_block(512, 1024)

    # Decoder layers
    self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
    self.att4 = AttentionBlock(512, 512, 256)
    self.decoder4 = self.conv_block(1024, 512)

    self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.att3 = AttentionBlock(256, 256, 128)
    self.decoder3 = self.conv_block(512, 256)

    self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.att2 = AttentionBlock(128, 128, 64)
    self.decoder2 = self.conv_block(256, 128)

    self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.att1 = AttentionBlock(64, 64, 32)
    self.decoder1 = self.conv_block(128, 64)

    # Final output layer
    self.final = nn.Conv2d(64, out_channels, kernel_size=1)

  def conv_block(self, in_channels, out_channels):
    """A basic convolutional block with Conv-BN-ReLU"""
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    # Encoding path
    e1 = self.encoder1(x)
    e2 = self.encoder2(nn.MaxPool2d(2)(e1))
    e3 = self.encoder3(nn.MaxPool2d(2)(e2))
    e4 = self.encoder4(nn.MaxPool2d(2)(e3))

    # Bottleneck
    b = self.bottleneck(nn.MaxPool2d(2)(e4))

    # Decoding path with attention
    d4 = self.up4(b)
    e4_att = self.att4(d4, e4)
    d4 = torch.cat((e4_att, d4), dim=1)
    d4 = self.decoder4(d4)

    d3 = self.up3(d4)
    e3_att = self.att3(d3, e3)
    d3 = torch.cat((e3_att, d3), dim=1)
    d3 = self.decoder3(d3)

    d2 = self.up2(d3)
    e2_att = self.att2(d2, e2)
    d2 = torch.cat((e2_att, d2), dim=1)
    d2 = self.decoder2(d2)

    d1 = self.up1(d2)
    e1_att = self.att1(d1, e1)
    d1 = torch.cat((e1_att, d1), dim=1)
    d1 = self.decoder1(d1)

    # Final output
    out = self.final(d1)
    return out

if __name__ == '__main__':
  # Define input tensor
  input_tensor = torch.randn(1, 1, 1024, 1024)  # Batch size = 1, single channel

  # Initialize the model
  model = AttentionUNetGenerator()

  # Forward pass
  output = model(input_tensor)

  # Check output shape
  print(f"Output shape: {output.shape}")  # Should be torch.Size([1, 1, 1024, 1024])
  print(f'Model: {model}')
  print(summary(model, (1,1,1024,1024)))
