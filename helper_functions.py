from imports import np, isfile, dataset_path, lab2rgb, rgb2lab, warn,virtual_memory, collect, remove
def convert_rgb_to_lab(image: np.ndarray, enable_assertions:bool=False) -> np.ndarray: 
    """
    Ova funkcija konvertuje image iz sRGB prostora u L*a*b* prostor.
    
    Parameters
    -----------
    image : np.NDArray, shape: Tuple[int,int,int[,int]]
        Slika(ili slike) treba da bude 3D/4D numpy array, gde je prva dimenzija redni broj slike(ako ih ima vise), druga dimenzija broj kanala, a treca i cetvrta sirina x visina. 
    
    Returns
    -------
        Returns a `np.ndarray` of shape (num_of_images, 3, height, width) if `image` is 4D, else if `image` is 3D it returns (3, height, width).
    """
    if enable_assertions:
        assert 3 in image.shape, f"Nije pronadjena nijedna dimenzija koja je =3"
        assert 3 <= image.ndim <= 4, f"Ocekivani broj dimenzija ulaznog parametra je izmedju 3 i 4(inclusive), a dobijeno je {image.ndim}"
        assert np.all( (image >= 0) & (image <= 255 )), f"Ocekivano da slike budu RGB formata s opsegom [0,255], to nije dobijeno ovde..."
        if np.all( (image == 0) | (image == 255)):
            warn(f"Postoje slike koje su iskljucivo bele ili crne")
    if image.ndim == 3:
        if enable_assertions:
            assert image.shape[0] == 3, f"Pogresna dimenzija na poziciji shape[0], ocekivano 3, dobijeno {image.shape[0]}"
        lab_image = rgb2lab(1.0/255*image, channel_axis=0)
        lab_image[1:] = lab_image[1:] / 128
        # X = lab_image[0]
        # Y = lab_image[1:]
        if enable_assertions:
            assert np.all( ( lab_image[0] >= 0) & ( lab_image[0] <= 100) ), f"U L* kanalu pronadjeno nedozvoljenih vrednosti"
            assert np.all( ( lab_image[1:] >= -1) & ( lab_image[1:] <= 1) ), f"U a*b* kanalima pronadjeno nedozvoljenih vrednosti"
            if np.all( (lab_image[0] == 0.0) | (lab_image[0] == 100.0) | (lab_image[1:] == 1.0) | (lab_image[1:] == -1.0 ) | (lab_image[1:] == 0.0 )):
                warn(f"Potencijalno postoje slike koje su iskljucivo bele ili crne")
    elif image.ndim == 4:
        if enable_assertions:
            assert image.shape[1] == 3, f"Pogresna dimenzija na poziciji shape[1], ocekivano 3, dobijeno {image.shape[1]}"
        lab_image = rgb2lab(1.0/255*image, channel_axis=1)
        lab_image[:,1:] = lab_image[:,1:] / 128
        if enable_assertions:
            assert np.all( (lab_image[:,0:1] >= 0 ) & (lab_image[:,0:1] <= 100 )), f"U L* kanalu pronadjeno nedozvoljenih vrednosti"
            assert np.all( (lab_image[:,1:] >= -1 ) & (lab_image[:,1:] <= 1 )), f"U a*b* kanalima pronadjeno nedozvoljenih vrednosti"
            if np.all( (lab_image[:,0:1] == 0.0) | (lab_image[:,0:1] == 100.0) | (lab_image[:,1:] == 1.0) | (lab_image[:,1:] == -1.0 ) | (lab_image[:,1:] == 0.0 ) ):
                warn(f"Potencijalno postoje slike koje su iskljucivo bele ili crne")
        # X = lab_image[:,0]
        # Y = lab_image[:,1:]
    return lab_image

def convert_lab_to_rgb(image: np.ndarray,enable_assertions:bool=False, denormalize=False) -> np.ndarray:
    """
    Ova funkcija konvertuje iz L*a*b* prostora u sRGB prostor.

    Parameters
    ----------
    image: np.NDArray, shape: Tuple[int,int,int[,int]]
        Slika(ili slike) treba da bude 3D/4D numpy array, gde je prva dimenzija redni broj slike(ako ih ima vise), druga dimenzija oznacava kanal(L*, a* ili b*), a treca i cetvrta sirina x visina. L* kanal mora da sadrzi vrednosti od 0 do 100, dok a* i b* moraju imati vrednosti izmedju -128 i 127.
    denormalize : boolean=False
        Da li denormalizovati podatke. Ako je denormalize=`True`, onda se koristi sRGB opseg [0,255], u protivnom se koristi [0,1]

    Returns
    -------
    Vraca nam sliku(ili slike) u `numpy.ndarray` formatu.   
    """
    if enable_assertions:
        assert 3 in image.shape, f"Nije pronadjena nijedna dimenzija koja je =3"
        assert 3 <= image.ndim <= 4, f"Ocekivani broj dimenzija ulaznog parametra je izmedju 3 i 4(inclusive), a dobijeno je {image.ndim}"
    if image.ndim == 3:
        if enable_assertions:
            assert image.shape[0] == 3, f"Pogresna dimenzija na poziciji shape[0], ocekivano 3, dobijeno {image.shape[0]}"
            assert np.all( (image[0] >= 0.0) & ( image[0] <= 100.0) & (image[1:] >= -128) & ( image[1:] <= 128) )
        rgb_image = lab2rgb(image, channel_axis=0)
    elif image.ndim == 4:
        if enable_assertions:
            assert image.shape[1] == 3, f"Pogresna dimenzija na poziciji shape[1], ocekivano 3, dobijeno {image.shape[1]}"
            assert np.all( (image[:,0:1] >= 0.0) & ( image[:,0:1] <= 100.0) & (image[:,1:] >= -128) & ( image[:,1:] <= 128) )
        rgb_image = lab2rgb(image, channel_axis=1)
    # rgb_image = rgb_image.reshape(1 if image.ndim == 3 else image.shape[0], 3, image.shape[1 if image.ndim == 3 else 2], image.shape[2 if image.ndim == 3 else 3])
        if enable_assertions:
            assert np.all( (rgb_image >= 0) & (rgb_image <=1.0)), f"Ocekivani opseg RGB vrednosti 0-1 je prekrsen"
            if np.all( (rgb_image == 0.0) | (rgb_image == 1.0)):
                warn(f"Postoje slike koje su iskljucivo crne ili bele")
    return rgb_image if not denormalize else (rgb_image * 255).astype(np.uint8)

def load_data(mmap_mode = None, percentage:int = 0.7, shape=(25000,3,224,224), returnJoinedInstead=False):
    """
    Treba da ucita podatke sa diska kao memory map. Memory-mapped podaci se ne ucitavaju svi u memoriju, vec se ucitavaju sa diska direktno po potrebi. 

    Parameters
    ----------
    mmap_mode : str | None
        U kom rezimu treba da ucitamo finalni dataset. Za vise videti [link](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html#numpy.memmap). Ako je `None`, ucitace ceo dataset u memoriju.
    percentage : int = 0.7
        Koliko procenata dostupne sistemske memorije zelimo iskoristiti za konverziju.
    
    Returns
    -------
        Ucitani L*a*b* dataset `np.ndarray`, ako je mmap_mode=`None`, u protivnom vraca `memmap` i cita se sa diska.
    """
    should_delete_dataset=False
    assert 0.1 < percentage < 0.9, f"Ocekivan opseg procenata [10%,90%], dobijeno {percentage}"
    if isfile(dataset_path + 'lab_dataset.npy'):
        return np.memmap(dataset_path + 'lab_dataset.npy',mode=mmap_mode, shape=shape,dtype="float16") if not returnJoinedInstead else np.load(dataset_path + 'joined_dataset.npy', mmap_mode='r')
    elif isfile(dataset_path + 'joined_dataset.npy'):
        joined_dataset = np.load(dataset_path + 'joined_dataset.npy', mmap_mode='r') # ucitavamo zdruzeni dataset
        dataset_shape = joined_dataset.shape
        dataset = np.memmap(dataset_path + 'lab_dataset.npy', mode="w+", shape=(dataset_shape[0],3,224,224), dtype="float16") # kreiramo memory-mapped fajl za finalni dataset (7 GB).
        available_system_memory_in_GBs = virtual_memory().available/1024**3
        how_much_memory_to_reserve_for_conversion_in_GBs = available_system_memory_in_GBs * percentage
        converted_image_size_in_GBs = dataset_shape[0] * 8 * 3 * 224 * 224 / 1024**3  # 28 GB u sustini ako sve odjednom konvertujem.
        print(f"Dostupna memorija za konverziju slika {how_much_memory_to_reserve_for_conversion_in_GBs}")
        print(f"Memorija potrebna za konverziju slika {converted_image_size_in_GBs}")

        try:
            if how_much_memory_to_reserve_for_conversion_in_GBs - converted_image_size_in_GBs >= 1: # Ostavljamo 1 GB overhead-a
                image = convert_rgb_to_lab(joined_dataset)
                dataset[:] = image
            else: # U protivnom koristimo batched obradu.
                batch_size = int( (how_much_memory_to_reserve_for_conversion_in_GBs-1)*dataset_shape[0] / (32*converted_image_size_in_GBs) ) # -1 zbog memory overheada. 32 jer rgb2lab koristi float64...
                print(f"Batch size:\t{batch_size}")
                assert batch_size > 1, f"Nemate dovoljno memorije za ovakvu operaciju, ocekivano je da batch_size bude veci od 1, ali je {batch_size}"
                for i in range(0, dataset_shape[0], batch_size+1):
                    dataset[i:i+batch_size] = convert_rgb_to_lab( joined_dataset[i:i+batch_size] )
                dataset[i:] = convert_rgb_to_lab( joined_dataset[i:] )
        except MemoryError as e:
            print(f"Doslo je do greske:\n{e}")
            should_delete_dataset = True
        except Exception as e:
            should_delete_dataset = True
            print(e)
        finally:
            dataset.flush()
            print("Flushed!")
            del dataset, joined_dataset
            collect()
            if should_delete_dataset:
                remove(dataset_path + 'lab_dataset.npy')
                return

        print("Kreirani finalni dataset")
        return load_data(mmap_mode, percentage, dataset_shape)
    else:
        X = np.load(dataset_path + 'l/gray_scale.npy',mmap_mode='r').reshape(25000,1,224,224)
        y_file_to_load = [ f'ab/ab/ab{i}.npy' for i in range(1,4)]
        Y = np.concatenate( [ np.load(dataset_path + file) for file in y_file_to_load ], axis=0 ).reshape(25000,2,224,224)
        np.save(dataset_path + 'joined_dataset.npy', np.concatenate( (X,Y), axis=1))
        del X,Y,y_file_to_load
        collect()
        print("Kreirani zdruzeni dataset")
        return load_data(mmap_mode, percentage)