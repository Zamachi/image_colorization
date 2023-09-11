from imports import torch
class CICAFFModel(torch.nn.Module):
    def __init__(self, number_of_classes: int=313):
        super(CICAFFModel, self).__init__()
        self.en1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,64,kernel_size=1,stride=1),
            torch.nn.ReLU(True)
        )
        self.en2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),# NOTE: mora bude stride=2
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # NOTE: mora bude stride=1
            torch.nn.ReLU(True)
        )
        self.en3 = EncoderBlock(128,256)
        self.en4 = EncoderBlock(256,512)
        self.en5 = EncoderBlock(512,1024)
        self.en6 = EncoderBlock(1024,2048)

        self.classification_subnet = ClassificationSubnetwork()
        #NOTE: po radu
        #aff1_out = AFF1(Subsample_4(en1_out), Sumbsample_2(en2_out), en3_out, Upsample_2(en4_out), Upsample_4(en5_out))
        self.aff1 = AFFModule(desired_multiplier=1/4, output_depth=256)
        #aff2_out = AFF2(Subsample_8(en1_out), Sumbsample_4(en2_out), Sumbsample_2(en3_out), en4_out, Upsample_2(en5_out))
        self.aff2 = AFFModule(desired_multiplier=1/8, output_depth=512)
        #aff2_out = AFF2(Subsample_16(en1_out), Sumbsample_8(en2_out), Sumbsample_4(en3_out), Sumbsample_2(en4_out), en5_out)
        self.aff3 = AFFModule(desired_multiplier=1/16, output_depth=1024)

        self.decode3 = DecoderModule(2048,1024)
        self.decode2 = DecoderModule(1024,512)
        self.decode1 = DecoderModule(512,256)
        self.output = torch.nn.Sequential(
            torch.nn.Conv2d(256,number_of_classes, kernel_size=1) # NOTE da li ovako treba da konstruisem sloj? Da li treba da bude dublja mreza? Ili mozda decode1 treba da da output [batch_size, 313, h/4, w/4]? Nejasno...
        )

    def forward(self, x):
        #NOTE reassignment vrv ne treba zbog ReLU(True)? proveriti...
        en1_out = self.en1(x) 
        en2_out = self.en2(x)
        en3_out = self.en3(x)
        en4_out = self.en4(x)
        en5_out = self.en5(x)
        aff1_out = self.aff1(en1_out, en2_out, en3_out, en4_out, en5_out)
        aff2_out = self.aff2(en1_out, en2_out, en3_out, en4_out, en5_out)
        aff3_out = self.aff3(en1_out, en2_out, en3_out, en4_out, en5_out)
        x_g = self.en6(x)
        image_category_probability = self.classification_subnet(x_g)
        decoder3_out = self.decode3(aff3_out,x_g)
        decoder2_out = self.decode2(aff2_out,decoder3_out)
        decoder1_out = self.decode1(aff1_out,decoder2_out)
        color_category_probability = torch.nn.Sigmoid()(self.output(decoder1_out))
        if self.training:
            return image_category_probability, color_category_probability
        else:
            return # vratiti recovered boju?
        

class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(EncoderBlock, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(True),
            ResidualBlock(in_channels, out_channels),
            torch.nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel:int, out_channel:int):
        super(ResidualBlock,self).__init__()
        assert in_channel * 2 == out_channel, f"Input channel provided was {in_channel}, expected output channel {in_channel*2}, got {out_channel}. Output channel has to be double the size of input channel."
        self.upper = torch.nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.lower1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.lower2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.residual_activation = torch.nn.ReLU()
    def forward(self, x):
        return  self.residual_activation(self.upper(x)) + self.residual_activation( self.lower2( self.residual_activation( self.lower1(x)))) 

class ClassificationSubnetwork(torch.nn.Module):
    def __init__(self):
        super(ClassificationSubnetwork,self).__init__()
        self.conv_module = torch.nn.Sequential(
            torch.nn.Conv2d(2048,1024, kernel_size=3, stride=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(1024,1000, kernel_size=3, stride=1),
            torch.nn.ReLU(True),
            torch.nn.AvgPool2d(3, 1),
            torch.nn.Softmax(dim=1),
        )
    def forward(self, x):
        return self.conv_module(x)

class AFFModule(torch.nn.Module):
    def __init__(self, desired_multiplier: float, output_depth: int, concatenated_tensor_size:int=1984, original_image_size:int=224):
        """
        Inicijalizuje AFF modul iz rada

        Parameters
        ----------
        desired_multiplier : float
            Multiplikator originalne dimenzije slike, tj. koliko se originalna slika umanjuje
        output_depth : int
            Koliko treba da iznosi dubina sloja prilikom izlaza iz AFF modula(treba da bude kompatibilna sa ulazom u dekoder)
        concatenated_tensor_size : int=1984,
            Posto se vrsi konkatenacija outputova, moramo znati njihovu zbirnu velicinu, pre nego sto propustimo konkatenirani tenzor kroz sloj. Treba da iznosi: `sum([input_tensor.channel_depth for input_tensor in input_tensors])`
        original_image_size : int = 224
            Veličina slike. Slika mora da bude kvadrat, što znači original_image_size = height = width, default je 224.
        """
        super(AFFModule,self).__init__()
        self.original_size = original_image_size
        self.desired_multiplier = desired_multiplier
        self.convolute_concatenated = torch.nn.Sequential(
            #TODO mozda treba dodati jos par slojeva dublje?
            torch.nn.Conv2d(concatenated_tensor_size, output_depth, kernel_size=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(output_depth, output_depth, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(True), 
        )
    def forward(self, *encoder_layers):
        return self.convolute_concatenated(torch.cat([self.transform(layer) for layer in encoder_layers],dim=1))
    
    def transform(self,  x:torch.Tensor):
        # NOTE posto nisu naveli kako se radi upsample/downsample, koristio sam interpolate koji nema learnable parametre u toku treninga
        return x if self.original_size * self.desired_multiplier == x.shape[2] else torch.nn.functional.interpolate(x, size=tuple([int(self.original_size*self.desired_multiplier)]*2))
    
class DecoderModule(torch.nn.Module):
    def __init__(self, in_channels: int, output_channels: int):
        super(DecoderModule, self).__init__()
        self.transpose = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, output_channels, kernel_size=2, stride=2),
            torch.nn.ReLU()
        )
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, output_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU()
        )

    def forward(self, aff_output, x):
        # NOTE ideje
        # 1. U raud kaze da se input u dekoderski blok prvo konkatenira sa aff_outputom, pa se onda progura kroz blok, a prvi ulaz u bloku je transponovana konvolucija. Ali ne mogu da se konkateniraju, jer su im rezolucije nekompatibilne... 
        # 2. Mozda se prvo x_g upskejluje, da bude [batch_size, 1024?(ili bilo koji drugi depth I guess), h/16, w/16]
            # 2.1 Pa onda oba konkaterniram, pa propustim kroz dekoderski blok
            # 2.2 Mozda propustim oba kroz transponovanu konvoluciju, nezavisno jedno od drugog, da ih svedem na odgovarajuce dimenzije, pa tek onda konkateniram, i tek onda kroz konvoluciju sa stride=1?
        # 3. Mozda moram i AFF output da svedem na [batch_size, 512, h/16, w/16], kao i x_g, pa tek onda konkatenacija, pa tek onda da propustim to kroz dekoderski blok?
        return self.conv(torch.cat([self.transpose(x), aff_output], dim=1))