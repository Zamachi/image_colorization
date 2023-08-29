from imports import torch
class SoftmaxAnnealedMean(torch.nn.Module):
    def __init__(self, temperature:float=0.38):
        super(SoftmaxAnnealedMean, self).__init__()
        self.temperature = temperature

    def forward(self, ground_truth, dim:int=1, isLogits:bool=True):
        """
        Sracunava annealed mean raspodele verovatnoca. Ovo koristimo da predvidjenu distribuciju mapiramo u tacku u ab prostoru.

        Parameters
        ----------
        ground_truth : torch.Tensor
            Istinitosne vrednosti za ab kanale oblika (batch_size, number_of_classes, width, height)
        dim : int
            Dimenzija po kojoj se vrsi sumiranje, ako je broj klasa na drugoj poziciji, koristiti defaultnu vrednost `dim=1`.
        isLogits : bool = True
            Da li su vrednosti u prosledjenom tenzoru logiti ili ne - ako jesu, konvertovace ih u verovatnoce automatski.
        """
        if isLogits:
            ground_truth = torch.nn.Softmax(dim=dim)(ground_truth)
        return torch.exp( torch.log(ground_truth) / self.temperature ) / torch.sum( torch.exp( torch.log(ground_truth) / self.temperature ) , dim=dim)