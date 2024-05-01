import torch

class UNet(torch.nn.Module):  # creates instance of Base class for all neural network modules
    """Takes in patches of 128^2 RGB, returns 88^2"""

    def __init__(self, out_channels=1, dropout_prob = 0.5):
        super().__init__()

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

        self.dropout = torch.nn.Dropout2d(p=dropout_prob)

        # Convenience
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):# dropout
        # Down, keeping layer outputs we'll need later.
        # print('-forward input', x.shape)
        x1 = self.relu(self.conv1A(x))
        x1_dropped = self.dropout(x1)
        l1 = self.relu(self.conv1B(x1_dropped))
        l1_dropped = self.dropout(l1)
        # print('-forward l1', l1.shape)

        x2 = self.relu(self.conv2A(self.pool(l1_dropped)))
        x2_dropped = self.dropout(x2)
        l2 = self.relu(self.conv2B(x2_dropped))
        l2_dropped = self.dropout(l2)
        # print('-forward l2', l2.shape)

        x3 = self.conv3A(self.pool(l2_dropped))
        x3_dropped = self.dropout(x3)
        out = self.relu(self.conv3B(self.relu(x3_dropped)))
        out = self.dropout(out)
        # print('-forward out', out.shape)

        # Up, now we overwritte out in each step.
        out = torch.cat([self.convtrans34(out), l2], dim=1)
        # print('-forward out cat1', out.shape)

        x4 = self.relu(self.conv4A(out))
        x4_dropped = self.dropout(x4)
        out = self.relu(self.conv4B(x4_dropped))
        out = self.dropout(out)
        # print('-forward out 4 layer', out.shape)

        out = torch.cat([self.convtrans45(out), l1], dim=1)
        # print('-forward out cat2', out.shape)

        x5 = self.relu(self.conv5A(out))
        x5_dropped = self.dropout(x5)
        out = self.relu(self.conv5B(x5_dropped))
        out = self.dropout(out)
        # print('-forward out relu2', out.shape)

        # Finishing
        out = self.convfinal(out)
        # print('-forward final convfinal', out.shape)
        return out
