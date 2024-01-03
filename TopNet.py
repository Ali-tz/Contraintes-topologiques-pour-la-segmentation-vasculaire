import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()


        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.Softmax(dim = out_channels)  # Assuming you want average pooling here 1ere Softmax (pas pool)
        )

            # self.conv = nn.Sequential(
            #     nn.Conv3d(in_channels, out_channels//2, kernel_size=3, stride=1, padding=1, bias=False),
            #     nn.BatchNorm3d(out_channels//2),
            #     nn.ReLU(inplace=True),
            #     nn.Conv3d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            #     nn.BatchNorm3d(out_channels),
            #     nn.ReLU(inplace=True)
            # )
    def forward(self, x):
        return self.conv(x)

class FinalDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalDoubleConv, self).__init__()


        # self.conv = nn.Sequential(
        #     nn.Conv3d(in_channels, out_channels//2, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm3d(out_channels//2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm3d(out_channels),
        #     nn.AvgPool3d(kernel_size=2, stride=2)  # Assuming you want average pooling here 1ere Softmax (pas pool)
        # )

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class TopNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128, 256]):
        super(TopNET, self).__init__()

        self.downs = nn.ModuleList()
        self.ups1 = nn.ModuleList()
        self.ups2 = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)

        # Down part of the TopNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 1st Up part of the TopNET
        for feature in reversed(features):
            self.ups1.append(nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2))
            self.ups1.append(DoubleConv(feature*2, feature))

        # 2nd Up part of the TopNET
        n = len(features)
        cpt = 0
        for feature in reversed(features):
            self.ups2.append(nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2))
            self.ups2.append(DoubleConv(feature*2, feature))
            cpt+=1
            if cpt == n:
                self.ups2.append(nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2))
                self.ups2.append(FinalDoubleConv(feature*2, feature))
                break

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x1 = self.bottleneck(x)
        # print(x1.shape)
        x2 = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # for idx in range(0, len(self.ups), 2):
        #     x = self.ups[idx](x)
        #     skip_connection = skip_connections[idx//2]

        #     if x.shape != skip_connection.shape:
        #         x = F.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)

        #     concat_skip = torch.cat((skip_connection, x), dim=1)
        #     x = self.ups[idx+1](concat_skip)

        x1 = self.move_up(x1, skip_connections, self.ups1)
        x2 = self.move_up(x2, skip_connections, self.ups2)
        # print("\n\n")
        # print(x1.shape)
        # print(x2.shape)
        return self.final_conv(x1), self.final_conv(x2)
    
    def move_up(self, x, skip_connections, ups):
        for idx in range(0, len(ups), 2):
            x = ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = ups[idx+1](concat_skip)
        return x

    
def test():
    x = torch.randn((3,1,160,160,160))
    # model = UNET(in_channels=1, out_channels=1)
    model = TopNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    # print(x.shape)
    # assert preds.shape == x.shape


if __name__ == "__main__":
    test()