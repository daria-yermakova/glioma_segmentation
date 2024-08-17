import torch

class UNet(torch.nn.Module):  # creates instance of Base class for all neural network modules
    """Takes in patches of 128^2 RGB, returns 88^2"""

    def __init__(self, name, img_size=64, in_channel=4, out_channels=1, dropout_prob = 0.5):
        super().__init__()
        self.name = name
        self.img_size = img_size
        self.in_channel = in_channel
        self.out_channel = out_channels
        self.dropout_prob = dropout_prob

        # Learnable
        self.conv1A = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.conv1B = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)

        self.conv2A = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv2B = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        self.conv3A = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3B = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        self.conv4A = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv4B = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        self.conv5A = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.conv5B = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)

        self.convfinal = torch.nn.Conv2d(in_channels=8, out_channels=out_channels, kernel_size=1)

        self.convtrans34 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)  # stride - step
        self.convtrans45 = torch.nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)

        # Convenience
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):# normal working way
        # Down, keeping layer outputs we'll need later.
        # print('-forward input', x.shape)
        l1 = self.relu(self.conv1B(self.relu(self.conv1A(x))))
        # print('-forward l1', l1.shape)
        l2 = self.relu(self.conv2B(self.relu(self.conv2A(self.pool(l1)))))
        # print('-forward l2', l2.shape)
        out = self.relu(self.conv3B(self.relu(self.conv3A(self.pool(l2)))))
        # print('-forward out', out.shape)

        # Up, now we overwritte out in each step.
        out = torch.cat([self.convtrans34(out), l2], dim=1)
        # print('-forward out cat1', out.shape)

        out = self.relu(self.conv4B(self.relu(self.conv4A(out))))
        # print('-forward out 4 layer', out.shape)

        out = torch.cat([self.convtrans45(out), l1], dim=1)
        # print('-forward out cat2', out.shape)

        out = self.relu(self.conv5B(self.relu(self.conv5A(out))))
        # print('-forward out relu2', out.shape)

        # Finishing
        out = self.convfinal(out)
        # print('-forward final convfinal', out.shape)
        return out