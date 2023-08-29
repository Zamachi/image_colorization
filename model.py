from imports import torch
class ColorizerModel(torch.nn.Module):
    def __init__(self, number_of_classes:int):
        super(ColorizerModel, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)
        self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
        self.conv7 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv8 = torch.nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv9 = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upsample1 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv10 = torch.nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv11 = torch.nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.upsample3 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        # self.last = nn.Conv2d(n_prev, number_of_classes, kernel_size=1, stride=1, padding=0)
        # self.model_output = nn.Conv2d(number_of_classes,2, kernel_size=1,padding=0,dilation=1,stride=1,bias=False)
        self.model_output = torch.nn.Conv2d(32,number_of_classes, kernel_size=1,padding=0,dilation=1,stride=1, bias=False)
        # self.softmax = SoftmaxAnnealedMean()
        # self.ab_output = nn.Conv2d(number_of_classes, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        # self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        x = torch.nn.ReLU()(self.conv1(x))
        x = torch.nn.ReLU()(self.conv2(x))
        x = torch.nn.ReLU()(self.conv3(x))
        x = torch.nn.ReLU()(self.conv4(x))
        x = torch.nn.ReLU()(self.conv5(x))
        x = torch.nn.ReLU()(self.conv6(x))
        x = torch.nn.ReLU()(self.conv7(x))
        x = torch.nn.ReLU()(self.conv8(x))
        x = torch.nn.ReLU()(self.conv9(x))
        x = self.upsample1(x)
        x = torch.nn.ReLU()(self.conv10(x))
        x = self.upsample2(x)
        x = torch.nn.ReLU()(self.conv11(x))
        # x = self.tanh(self.conv12(x))
        x = self.upsample3(x)
        x = self.model_output(x)
        return x
        # if not isInference:
        #     return x
        # else:
        #     return self.upsample4(self.ab_output(self.softmax(x)))