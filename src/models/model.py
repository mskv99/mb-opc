import torch.nn as nn

class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, leaky=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class deconv_block(nn.Module):
    """
    Deconvolution Block
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, output_padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()

        n1 = 32
        filters = [n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32] #[64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_head = conv_block(in_ch, n1, kernel_size=7, stride=1, padding=3)

        self.conv0 = conv_block(n1, filters[0], stride=2)         #conv_block(32, 64, stride)
        self.conv1 = conv_block(filters[0], filters[1], stride=2) #conv_block(64,128)
        self.conv2 = conv_block(filters[1], filters[2], stride=2) #conv_block(128, 256)
        self.conv3 = conv_block(filters[2], filters[3], stride=2) #conv_block(256, 512)
        self.conv4 = conv_block(filters[3], filters[4], stride=2) #conv_block(512, 1024)

        self.res0 = conv_block(filters[4], filters[4], stride=1) #conv_block()
        self.res1 = conv_block(filters[4], filters[4], stride=1)
        self.res2 = conv_block(filters[4], filters[4], stride=1)
        self.res3 = conv_block(filters[4], filters[4], stride=1)
        self.res4 = conv_block(filters[4], filters[4], stride=1)
        self.res5 = conv_block(filters[4], filters[4], stride=1)
        self.res6 = conv_block(filters[4], filters[4], stride=1)
        self.res7 = conv_block(filters[4], filters[4], stride=1)
        self.res8 = conv_block(filters[4], filters[4], stride=1)

        self.deconv0 = deconv_block(filters[0], n1, stride=2)
        self.deconv1 = deconv_block(filters[1], filters[0], stride=2)
        self.deconv2 = deconv_block(filters[2], filters[1], stride=2)
        self.deconv3 = deconv_block(filters[3], filters[2], stride=2)
        self.deconv4 = deconv_block(filters[4], filters[3], stride=2)

        self.conv_tail = nn.Conv2d(n1, out_ch, kernel_size=7, stride=1, padding=3)


    def forward(self, x):
        x_head = self.conv_head(x)

        x0_0 = self.conv0(x_head)
        x1_0 = self.conv1(x0_0)
        x2_0 = self.conv2(x1_0)
        x3_0 = self.conv3(x2_0)
        x4_0 = self.conv4(x3_0)

        xres = self.res0(x4_0)
        xres = self.res1(xres)
        xres = self.res2(xres)
        xres = self.res3(xres)
        xres = self.res4(xres)
        xres = self.res5(xres)
        xres = self.res6(xres)
        xres = self.res7(xres)
        xres = self.res8(xres)

        x4_1 = self.deconv4(xres)
        x3_1 = self.deconv3(x4_1)
        x2_1 = self.deconv2(x3_1)
        x1_1 = self.deconv1(x2_1)
        x0_1 = self.deconv0(x1_1)

        output = self.conv_tail(x0_1)
        return output
