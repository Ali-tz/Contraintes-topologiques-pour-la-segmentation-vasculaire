import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F


# class DoubleConv(nn.Module):

#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False), # input size = output size
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace = True),
#             nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False), # input size = output size
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace = True)
#         )

#     def forward(self, x):
#         return self.conv(x)

# class UNET(nn.Module):
#     def __init__(self, in_channels = 3, out_channels = 1, features = [64, 128, 256, 512 ]) :
#         super(UNET, self).__init__()

#         self.downs = nn.ModuleList()
#         self.ups = nn.ModuleList()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)

#         # Down part of the UNET
#         for feature in features:
#             self.downs.append(DoubleConv(in_channels, feature))
#             in_channels = feature

#         # Up part of the UNET
#         for feature in reversed(features):
#             self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2,))
#             self.ups.append(DoubleConv(feature*2, feature))
        
#         self.bottleneck = DoubleConv(features[-1], features[-1]*2)
#         self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    
#     def forward(self, x):
#         skip_connections = []

#         for down in self.downs:
#             x = down(x)
#             skip_connections.append(x)
#             x = self.pool(x)

#         x = self.bottleneck(x)
#         skip_connections = skip_connections[::-1]

#         for idx in range(0, len(self.ups), 2):
#             x = self.ups[idx](x)
#             skip_connection = skip_connections[idx//2]

#             if x.shape != skip_connection.shape :
#                 x = TF.resize(x, size = skip_connection.shape[2:])

#             concat_skip = torch.cat((skip_connection,x),dim = 1)
#             x = self.ups[idx+1](concat_skip)

#         return self.final_conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()


        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.AvgPool3d(kernel_size=2, stride=2)  # Assuming you want average pooling here 1ere Softmax (pas pool)
        )

    def forward(self, x):
        return self.conv(x)

# Example usage:
# my_double_conv = DoubleConv(in_channels=3, out_channels=64, last=True)


# Après le embedding space, faire une conversion d'une image de segmentation à une classififcation avec edt scipy


class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128, 256]):
        super(UNET, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)

        # Down part of the UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of the UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        print(x.shape)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        print("\n\n")
        print(x.shape)
        print("\n\n")
        return self.final_conv(x)
    




    
def test():
    x = torch.randn((3,1,160,160,160))
    model = UNET(in_channels=1, out_channels=1)
    # model = TopNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()

