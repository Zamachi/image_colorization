from imports import torch, np, gaussian_filter1d, cross_entropy
class ReweightedCrossEntropy(torch.nn.Module):
    def __init__(self, Q_value:int, class_weights_probs:np.ndarray, device, lambda_value=None, sigma_value=None):
        super(ReweightedCrossEntropy, self).__init__()
        self.lambda_value = 0.5 if lambda_value is None else lambda_value
        self.sigma_value = 5 if sigma_value is None else sigma_value
        self.Q_value = Q_value
        self.class_weights_probs = class_weights_probs
        self.device =device
        self.smoothed_distribution = gaussian_filter1d(self.class_weights_probs, sigma=self.sigma_value)
        self.w_value = ( (1-self.lambda_value) * self.smoothed_distribution + (self.lambda_value/self.Q_value) ) ** -1

    def forward(self, model_predictions: torch.Tensor, ground_truth: torch.Tensor):
        """
        Metoda uzima predikcije modela, koji su nenormalizovani logiti, oblika (N, C, d1, d2 ... dK), gde je K>=1, gde je C broj klasa, a N batch size i daje reweighted cross entropy loss i istinitosne vrednosti oblika (N,d1,d2 ... dK), gde je K>=1, gde je svaka vrednost u opsegu [0,C). 

        Parameters
        ----------
        model_predictions : torch.Tensor
            Raw predikcije koje daje model(nenormalizovani logiti), oblika (N, C, d1, d2 ... dK), gde je K>=1
        ground_truth : torch.Tensor
            Konkretan tenzor klasa za dati piksel, sadrzi **indekse** klasa, tj. svaka vrednost je [0,C), oblika (N, d1, d2 ... dK), gde je K>=1. 
        """

        return cross_entropy(model_predictions,ground_truth,torch.as_tensor(self.w_value, device=self.device, dtype=torch.float32))