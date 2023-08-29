from helper_functions import load_data, convert_lab_to_rgb 
from imports import torch, islice
class IterableImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_indices: list, num_of_classes:int, mappings:dict, return_joined:bool=False, transform=None, bins=None):
        """
        Inicijalizuje dataset

        Parameters
        ----------
        data_indices : list
            Koje redove iz dataseta koristimo
        transform(optional) : torchvision.Compose
            Transformacije koje koje primenjujemo nad nasim slikama.
        mappings : dict
            Mapiranja na za 
        return_joined : bool=False
            Da li da se vrati joined dataset ili konvertovani, ako je False onda se vraca Lab DS.
        transform = None
            Da li specificirati neke transformacije
        bins: np.ndarray
            Rastuci niz np.ndarray nizova
        """
        # assert data.ndim == 4, f"Ocekivan broj dimenzija dataseta 4, dobijeno {data.ndim}"
        # assert data.shape == (data.shape[0],3,224,224), f"Ocekivan shape dataseta (25000,3,224,224), dobijeno {data.shape}"
        # assert np.all( (data >= 0) & (data <= 255)), f"Slike treba da su u RGB formatu!"
        self.data = load_data("r", returnJoinedInstead=return_joined) 
        self.is_joined = return_joined
        self.indices = dataset_indices
        self.transform = transform
        self.num_of_classes = num_of_classes
        self.mappings = torch.as_tensor(mappings.copy())
        self.bins = torch.as_tensor(bins.copy())

    def __len__(self):
        return len(self.data)

    def process_data(self, dataset):
        for idx in self.indices:
            img = dataset[idx].copy()
            if self.transform:
                # if self.is_joined:
                #     img_transformed = convert_rgb_to_lab(img_transformed)
                img_transformed = self.transform(torch.as_tensor( convert_lab_to_rgb(img) if not self.is_joined else img,dtype=torch.float32))

                yield img_transformed[0].reshape(-1,224,224), self.map_ab_to_quantized(img_transformed[1:2] * 110 , self.mappings, bins=self.bins)
            else:
                # if self.is_joined:
                img = torch.as_tensor( convert_lab_to_rgb(img) if not self.is_joined else img,dtype=torch.float32)
                yield img[0].reshape(-1,224,224), self.map_ab_to_quantized(img[1:2] * 110, self.mappings, bins=self.bins)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info() # NOTE ako je None, onda je single-process
        print(worker_info)
        if worker_info is None:
            return iter(self.process_data(self.data))
            # return islice(map(self.process_data, iter(self.data)),0)
        total_number_of_workers = worker_info.num_workers
        worker_id = worker_info.id
        dataset_length = self.__len__()

        mapped_iterator = map(self.process_data, iter(self.data))
        # return islice(iter(self.process_data(self.data[self.indices])), worker_id, None, total_number_of_workers)
        return islice(mapped_iterator, worker_id, None, total_number_of_workers)
 
    #   lab_image = rgb2lab( (np.array(sth['image'].resize((224,224))).reshape(224,224,3))/255, channel_axis=2)
    def map_ab_to_quantized(self,ab_image, mappings: torch.Tensor, bins: torch.Tensor):
        ab_joint = torch.bucketize(ab_image, boundaries=bins, right=True).reshape(-1,224,224)
        #         #WARNING: problem u mapiranju je mozda taj sto su date koordinate centri kvadratica(bina), sto znaci da merimo +/- binsize/2 oko tog
        result = ((abs(mappings.reshape(len(mappings),2,1,1)) - abs(ab_joint)) ** 2).sum(dim=1).argmin(dim=0)
        return result