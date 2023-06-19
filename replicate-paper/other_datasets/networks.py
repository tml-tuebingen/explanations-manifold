import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        
        if stride == 16:
            blocks = [
                nn.Conv2d(in_channel, channel//2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//2, channel//2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//2, channel//2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]
        
        elif stride == 8:
            blocks = [
                nn.Conv2d(in_channel, channel//2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//2, channel//2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]
            
        elif stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel//2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel//2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 16:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel//2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel//2, channel//2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel//2, channel//2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel//2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )
        
        elif stride == 8:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel//2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel//2, channel//2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel//2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )
            
        elif stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel//2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel//2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)
    
class AutoEncoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        
        self.E = Encoder(in_channel, channel, n_res_block, n_res_channel, stride)
        self.D = Decoder(channel, out_channel, channel, n_res_block, n_res_channel, stride)
    
    def forward(self, x):
        z = self.E(x)
        x_rec = self.D(z)
        return x_rec, z
    
    def encode(self, x):
        return self.E(x)
    
    def decode(self, z):
        return self.D(z)
    
    
    
class ConvAE(nn.Module):
    def __init__(self, image_channels, init_channels, kernel_size, latent_dim):
        super(ConvAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels*16, kernel_size=kernel_size, 
            stride=1, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*16, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec5 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
    
    def encode(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        return mu
    
    def decode(self, z):
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        reconstruction = torch.sigmoid(self.dec5(x))
        return reconstruction
        
    def forward(self, x):
        # parameters of the posterior distribution
        mu = self.encode(x)
        # sample from the posterior, with reparameterization trick
        z = mu
        # decode
        reconstruction = self.decode(z)
        return reconstruction, z
    
# M1 = Encoder(3, 8, 4, 32, 16)
# x = torch.randn(4,3,224,256)
# y = M1(x)
# print(y.shape)
# M2 = Decoder(8, 3, 8, 4, 32, 16)
# y1 = M2(y)
# print(y1.shape)

# M3 = AutoEncoder(3, 3, 8, 4, 32, 16)
# y, z = M3(x)
# print(y.shape, z.shape)