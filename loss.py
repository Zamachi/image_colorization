from imports import torch
class CICCrossEntropyLoss(torch.nn.Module):
    def __init__(self, lambda_col: float=1, lambda_cls: float=0.003):
        """
        Definise CrossEntropy loss objekat koji sracunava objedinjen cross-entropy u skladu sa naucnim radom.

        Parameters
        ----------
        lambda_col : float = 1
            cross-entropy loss za kolorizacioni deo mreze
        lambda_cls : float = 0.0003
            hiperparametar definisan pri inicijalizaciji loss-a, tezinski faktor za `L_cls`.
        """
        super(CICCrossEntropyLoss,self).__init__()
        self.lambda_col = lambda_col
        self.lambda_cls = lambda_cls

    def forward(self, true_category_label: torch.Tensor, predicted_category_distribution: torch.Tensor, predicted_color_distribution: torch.Tensor, true_color_class: torch.Tensor):
        """
        Vraca loss iz rada definisan kao: lambda_col * L_col + lambda_cls * L_cls.
        ---------------------------------------------------------------------------
        * L_col - cross-entropy loss za kolorizacioni deo mreze
        * lambda_col - hiperparametar definisan pri inicijalizaciji loss-a, tezinski faktor za `L_col`.
        * L_cls - cross-entropy loss za klasifikacionu podmrezu
        * lambda_cls - hiperparametar definisan pri inicijalizaciji loss-a, tezinski faktor za `L_cls`.

        Parameters
        ----------
        true_category_label
            Ground truth za kategoriju slike
        predicted_category_distribution : torch.Tensor
            Raspodela verovatnoca za kategoriju slike koju predvidja klasifikaciona podmreza.
        true_color_class
            Ground truth za klasu boje piksela
        predicted_color_distribution : torch.Tensor
            Raspodela verovatnoca za kategoriju slike koju predvidja kolorizacioni deo.

        Shape:
            - true_category_label: Shape :math:`(N)` ili :math:`(N,1)`, ili :math:`(N,1,1)`.
            - predicted_category_distribution: Shape :math:`(N,1000)` ili :math:`(N,1000,1)`, ili :math:`(N,1000,1,1)`.
            - true_color_class: Shape
            - predicted_color_distribution: Shape
        """
        return self.lambda_col*self.colorization_loss(true_color_class, predicted_color_distribution) + self.lambda_cls * self.classification_subnetwork_loss(true_category_label,predicted_category_distribution)
    
    def classification_subnetwork_loss(self,true_category_label: torch.Tensor, predicted_category_distribution: torch.Tensor):
        # Compute the cross-entropy loss using the provided formula
        # loss = -torch.sum(Y_h_w_m * torch.log(Y_prime_h_w_m)) / n # NOTE ovo bi trebalo da je obicna 
        return torch.sum(true_category_label * torch.log(predicted_category_distribution).gather(dim=1,index=true_category_label.reshape(-1,1)))

    def colorization_loss(self,true_color_class, predicted_color_distribution, balance_weight_matrix):
        # return torch.sum(true_category_label * torch.log(predicted_category_distribution).gather(dim=1,index=true_category_label.reshape(-1,1)))
        return torch.sum( true_color_class * torch.log(predicted_color_distribution).gather(dim=1,index=true_color_class.reshape(-1,1)))


