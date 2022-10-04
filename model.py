import functools
import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ImageEmbeddings(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(ImageEmbeddings, self).__init__()
        self.act_fn = torch.nn.functional.relu

        self.conv0 = nn.Conv2d(in_size, hidden_size // 8, kernel_size=7, stride=2, padding=3, bias=True)
        self.norm0 = nn.BatchNorm2d(hidden_size // 8)
        self.conv1 = nn.Conv2d(hidden_size // 8, hidden_size // 4, kernel_size=3, stride=2, padding=1, bias=True)
        self.norm1 = nn.BatchNorm2d(hidden_size // 4)
        self.conv2 = nn.Conv2d(hidden_size // 4, hidden_size // 2, kernel_size=3, stride=2, padding=1, bias=True)
        self.norm2 = nn.BatchNorm2d(hidden_size // 2)
        self.conv3 = nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=3, stride=2, padding=1, bias=True)
        self.norm3 = nn.BatchNorm2d(hidden_size)

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.act_fn(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act_fn(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act_fn(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act_fn(x)
        return x


class ImageGenerationHead(nn.Module):
    def __init__(self, out_size, hidden_size):
        super(ImageGenerationHead, self).__init__()
        self.act_fn = torch.nn.functional.relu
        self.pad = nn.ReflectionPad2d(3)
        self.sigmoid = nn.Sigmoid()

        self.upconv0 = nn.ConvTranspose2d(hidden_size, hidden_size // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.norm0 = nn.BatchNorm2d(hidden_size // 2)
        self.upconv1 = nn.ConvTranspose2d(hidden_size // 2, hidden_size // 4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.norm1 = nn.BatchNorm2d(hidden_size // 4)
        self.upconv2 = nn.ConvTranspose2d(hidden_size // 4, hidden_size // 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.norm2 = nn.BatchNorm2d(hidden_size // 8)
        self.upconv3 = nn.ConvTranspose2d(hidden_size // 8, hidden_size // 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.norm3 = nn.BatchNorm2d(hidden_size // 16)
        self.conv4 = nn.Conv2d(hidden_size // 16, out_size, kernel_size=7, bias=True)

    def forward(self, x):
        x = self.upconv0(x)
        x = self.norm0(x)
        x = self.act_fn(x)
        x = self.upconv1(x)
        x = self.norm1(x)
        x = self.act_fn(x)
        x = self.upconv2(x)
        x = self.norm2(x)
        x = self.act_fn(x)
        x = self.upconv3(x)
        x = self.norm3(x)
        x = self.act_fn(x)
        x = self.pad(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        return x


class Model(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.out_size = out_size

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # image embedding
        self.image_embedding = ImageEmbeddings(self.in_size, self.hidden_size)

        # image generation
        self.image_generation = ImageGenerationHead(self.out_size, self.hidden_size)

        resnet = []
        for i in range(n_blocks):
            resnet += [ResnetBlock(self.hidden_size, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.resnet = nn.Sequential(*resnet)

    def forward(self, input):
        image_embedding = self.image_embedding(input)
        res_out = self.resnet(image_embedding)
        image_out = self.image_generation(res_out)
        return image_out