import pretty_errors
from imports import isfile, makedirs, exists, zipfile, dataset_path, model_weights_path, np, torch, Lion, lr_scheduler, DataLoader, FrechetInceptionDistance, collect
from helper_functions import load_data
from dataset import IterableImageDataset
from model import ColorizerModel
from reweighted_cross_entropy import ReweightedCrossEntropy
from softmax_annealed_mean import SoftmaxAnnealedMean
from training import fit

DEBUG=False
if DEBUG:
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if not isfile('./archive.zip'):
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('shravankumar9892/image-colorization',path='.',unzip=False)

if not exists(dataset_path):
    makedirs(dataset_path, exist_ok=True)
if not exists(model_weights_path):
    makedirs(model_weights_path, exist_ok=True)

if not exists(dataset_path+'ab') and not exists(dataset_path+'l'):
    with zipfile.ZipFile('./archive.zip', 'r') as zip_ref:
        zip_ref.extractall(dataset_path)

def main():
    #NOTE: ovo su stvari sa naucnog rada
    probs = np.load('./prior_probs.npy', mmap_mode='r')
    mappings = np.load('./pts_in_hull.npy', mmap_mode='r')
    num_classes = len(probs)
    ab_range = np.arange(-110, 120, 10)
    #NOTE: ovo su stvari sa naucnog rada

    test_size = 0.2
    validation_size = 0.15
    dataset_memory_map = load_data('r')
    test_length = int(test_size * dataset_memory_map.shape[0])
    remaining_length = dataset_memory_map.shape[0] - test_length
    validation_length = int(remaining_length * validation_size)
    training_length = remaining_length - validation_length

    indices = [i for i in range(dataset_memory_map.shape[0])]

    training_indices = np.random.choice(range(0, dataset_memory_map.shape[0]), size=training_length, replace=False)
    validation_indices = np.random.choice(list(set(range(0, dataset_memory_map.shape[0])) - set(training_indices)), size=validation_length, replace=False)
    test_indices = np.random.choice(list(set(range(0, dataset_memory_map.shape[0])) - set(training_indices) - set(validation_indices)), size=test_length, replace=False)

    # training_ds = IterableImageDataset(dataset_memory_map[training_indices], transform=transform)
    # validation_ds = IterableImageDataset(dataset_memory_map[validation_indices], transform=transform)
    training_ds = IterableImageDataset(training_indices, mappings=mappings, num_of_classes=num_classes, bins=ab_range)
    validation_ds = IterableImageDataset(validation_indices, mappings=mappings, num_of_classes=num_classes, bins=ab_range)

    number_of_epochs = 25
    batch_size = 32 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ColorizerModel(number_of_classes=num_classes).to(device)
    optimizer = Lion(model.parameters(), lr=3.67*0.001)
    learning_rate_scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=number_of_epochs * len(training_ds), eta_min=0.01)
    # optimizer = torch.optim.RMSprop(model.parameters())
    # learning_rate_scheduler = lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.9,end_factor=1/5,total_iters=number_of_epochs * len(training))

    for state in optimizer.state.values():
        for k,v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    loss = ReweightedCrossEntropy(Q_value=num_classes,class_weights_probs=probs,device=device)
    #NOTE: menjaj ovo ako hoces da ucitas weightove
    use_pretrained_weights = False 
    # training = DataLoader(training_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    # validation = DataLoader(validation_ds, batch_size=int(batch_size/4), shuffle=True, pin_memory=True)
    training = DataLoader(training_ds, batch_size=batch_size, pin_memory=True)
    validation = DataLoader(validation_ds, batch_size=batch_size, pin_memory=True)
    metric = FrechetInceptionDistance(feature=64, reset_real_features=False)
    epoch_start=0
    loss_start=None
    best_result = None
    if use_pretrained_weights:
        checkpoint = torch.load(model_weights_path+"checkpoint.tar", map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint[f'optimizer_{optimizer.__class__.__name__}'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        learning_rate_scheduler.load_state_dict(checkpoint[f'scheduler_{learning_rate_scheduler.__class__.__name__}'])
        # for state in learning_rate_scheduler.state_dict().values():
        #     for key, value in state.items():
        #         if isinstance(value, torch.Tensor):
        #             state[key] = value.to(device)
        epoch_start = checkpoint['epoch']
        loss_start = checkpoint['loss'] 
        best_result = checkpoint['best']
        model.eval()
        # model.load_state_dict(torch.load(model_weights_path+ 'model_weights.pth'))
        # optimizer.load_state_dict(torch.load(model_weights_path + f'optimizer_{optimizer.__class__.__name__}.pt', map_location=device))
        # learning_rate_scheduler.load_state_dict(torch.load(model_weights_path + f'scheduler_{learning_rate_scheduler.__class__.__name__}.pt', map_location=device))
    del training_ds, validation_ds, dataset_memory_map
    collect()
    torch.cuda.empty_cache()
    fit(
        model=model,
        optimizer=optimizer,
        training=training, 
        validation=validation,
        scheduler=learning_rate_scheduler,
        epochs=number_of_epochs,
        device=device, 
        gradient_accumulation_steps=12, 
        metric_for_early_stopping=metric, 
        enable_early_stopping=True,
        early_stopping_mode='min',
        patience=4,
        epoch_start=epoch_start, 
        loss=loss_start, 
        shouldEvaluate=False, 
        best=best_result, 
        loss_fn=loss)

if __name__ == '__main__':
    main()