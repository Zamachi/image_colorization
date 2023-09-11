from imports import np, torch
#NOTE empirical_probabilities moze da bude ucitan spolja i prosledjen kao parametar prilikom kreiranja I guess...A mozda mozemo i ovde da ucitamo?
class ColorCategoryModule():
    def __init__(self, empirical_probabilities:np.ndarray, number_of_color_categories:int=313, lambda_weight: float=0.5):
        self.empirical_probabilities = empirical_probabilities
        self.number_of_color_categories = number_of_color_categories
        self.lambda_weight = lambda_weight
    
    def __call__(self, x_a_b):
        return self.color_category(), self.balance_categories()
    
    def balance_categories(self):
        """
        The category balance module obtains the corresponding balance weight w(Zh,w) based on the color category Zh,w. NOTE: Zh,w je vrv isto sto i q.
        
        """
        omega = ( self.empirical_probabilities * ( 1 - self.lambda_weight) + (self.lambda_weight/self.number_of_color_categories) ) ** -1
        print("Omega:\t",omega)
        print("Omega(shape):\t",omega.shape)
        #NOTE: kako ovo treba da bude matrica, ne razumem?
        #NOTE zasto radimo omega ** -1 po formuli, kad ce to svakako obrisati ^-1 od pre?
        return (omega ** -1 ) / np.sum( (self.empirical_probabilities * omega),axis=0)  

    def color_category(self):
        return