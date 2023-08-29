from imports import torch, DataLoader, trange, time, collect, math, model_weights_path
from helper_functions import convert_lab_to_rgb
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False, also_use_timer=False, seconds_to_terminate:int=60*60):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self.also_use_timer=also_use_timer
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

        if also_use_timer:
            self.start_time=time.perf_counter()
            self.end_time = 0
            self.time_compare = lambda start,end: end-start >= seconds_to_terminate # NOTE Terminate after an hour
        else:
            self.start_time=None
            self.end_time=None
            self.time_compare = lambda start,end: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.is_tensor(metrics):
            if torch.isnan(metrics):
                return True
        elif type(metrics) == float and math.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print('terminating because of early stopping!')
            return True

        return False

    def time_ran_out(self):
        if self.time_compare(self.start_time, self.end_time):
            print("Terminating because of training time limit.")
            return True
        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


def evaluate(model: torch.nn.Module, validation: DataLoader, device, metric, is_called_from_training=False):
    model.eval()
    fid = [] 
    for step, batch in enumerate(validation):
        with torch.no_grad():
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            metric.update(torch.as_tensor(convert_lab_to_rgb(torch.cat( (inputs,targets*128), dim=1).cpu().numpy(),denormalize=True),dtype=torch.uint8), real=True)
            metric.update(torch.as_tensor(convert_lab_to_rgb(torch.cat( (inputs,outputs*128), dim=1).cpu().numpy(),denormalize=True),dtype=torch.uint8), real=False)
            fid.append(metric.compute().item())
    model.train()
    del batch
    collect()
    torch.cuda.empty_cache()
    return fid

def fit(model: torch.nn.Module, optimizer: torch.optim.Optimizer, training:DataLoader, validation: DataLoader, scheduler: torch.optim.lr_scheduler.LRScheduler, metric_for_early_stopping,  epochs:int=50, loss_fn=torch.nn.MSELoss(), gradient_accumulation_steps:int=8,enable_early_stopping:bool=True,patience:int=7,early_stopping_mode:str='min',delta_for_early_stopping:float=0,best:float=None,also_use_timer_for_early_stopping:bool=False, seconds_for_early_stopping:int=60*60, device:str='cpu', epoch_start:int=0, loss=None, shouldEvaluate:bool = True):
    """
    Trenira/fituje model.

    Parameters
    ----------
    model : nn.Module
        Ovo je objekat instanciranog modela kojeg treniramo
    optimizer : optim.Optimizer
        Optimizator parametara `model` koje koristimo
    training : DataLoader
        DataLoader za trening
    validation : DataLoader
        DataLoader za validaciju
    epochs : int, optional
        Broj epoha prilikom treninga
    loss_fn : optional
        Funkcija za generisanja loss-a tokom treninga `model`-a.
    scheduler : LRSCheduler 
        Scheduler za `learning_rate` 
    gradient_accumulation_steps : int, optional
        Koliko step-ova akumuliramo gradijente pre nego sto uradimo apdejt vejtova. Ako ne zelimo akumuliranje gradijenata, setovati ovaj parametar na 1.
    early_stopping_mode : str, optional
        Rezim rada early stopping mehanizma(moze biti `min` ili `max`)
    patience : int, optional
        Koliko koraka u EarlyStoppingu tolerisemo pre nego sto prekinemo trening
    delta_for_early_stopping : float, optional
        Tolerancija odstupanja performansi za early stopping
    metric_for_early_stopping : str, optional
        Koju metriku cemo koristiti za early stopping. 
    best : float, optional
        Najbolji rezultat koji je model postigao. Podrazumevano nema, ako instanciramo model od 0.
    also_use_timer_for_early_stopping : bool, optional
        Da li se koristi i tajmer za early stopping(ako npr. zelimo da trening traje odredjeno vreme)
    device : {'cpu', 'cuda'}
        Na kojem uredjaju zelimo da se vrsi trening.
    epoch_start : int=0
        Od koje epohe poceti trening, koristi se samo ako nastavljamo od checkpointa.
    loss 
        Koji je loss bio na poslednjoj sacuvanoj epohi?
    """
# model = model.to(device)
    trainingSteps = epochs * len(training)

    if enable_early_stopping:
        earlyStopping = EarlyStopping(patience=min(epochs, patience), mode=early_stopping_mode, min_delta=delta_for_early_stopping,also_use_timer=also_use_timer_for_early_stopping, seconds_to_terminate=seconds_for_early_stopping)
    best = best

    progress_bar = trange(trainingSteps)

    model.train()
    completed_steps = 0
    # Ovo navodno ubrzava
    torch.backends.cudnn.benchmark = True
    """
    Reason for turning off above:

    Setting torch.backends.cudnn.benchmark = True before the training loop can accelerate the computation. Because the performance of cuDNN algorithms to compute the convolution of different kernel sizes varies, the auto-tuner can run a benchmark to find the best algorithm (current algorithms are [these](https://github.com/pytorch/pytorch/blob/dab5e2a23ed387046d99f825e0d9a45bd58fccaa/aten/src/ATen/native/cudnn/Conv_v7.cpp#L268-L275), [these](https://github.com/pytorch/pytorch/blob/dab5e2a23ed387046d99f825e0d9a45bd58fccaa/aten/src/ATen/native/cudnn/Conv_v7.cpp#L341-L346), and [these](https://github.com/pytorch/pytorch/blob/dab5e2a23ed387046d99f825e0d9a45bd58fccaa/aten/src/ATen/native/cudnn/Conv_v7.cpp#L413-L418)). It's recommended to use turn on the setting when your input size doesn't change often. If the input size changes often, the auto-tuner needs to benchmark too frequently, which might hurt the performance. It can speed up by 1.27x to 1.70x for forward and backward propagation [ref](https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf).    
    """ 
    fid_over_training = dict()
    losses = []
    loss_now = loss
    loss_before = None
    for epoch in range(epoch_start, epochs):
        for step, batch in enumerate(training, start=1):
            # outputs = model(**batch)
            # loss = outputs.loss
            inputs, targets = batch
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            # print(inputs)
            loss = loss_fn(outputs, targets)
            loss_before = loss_now
            loss_now = loss
            losses.append(loss.item())
            loss = loss / gradient_accumulation_steps
            loss.backward()
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True) # set_to_none ne okida memset funkciju i navodno je to brze
                completed_steps += 1
        progress_bar.update(1)
        evaluation = evaluate(model, validation, device, metric_for_early_stopping, is_called_from_training=True) if shouldEvaluate else best
        if shouldEvaluate:
            evaluation = sum(evaluation) / len(evaluation)
            fid_over_training[epoch] = evaluation
        print(f"Tokom epohe {epoch+1} loss je bio {loss.item()} sa akumuliranjem, tj. {loss.item()*gradient_accumulation_steps} bez akumuliranja gradijenta, learning rate je {scheduler.get_last_lr()}, a evaluacija je dala metriku {evaluation}")
        del batch
        collect()
        torch.cuda.empty_cache() 
        if enable_early_stopping:
            earlyStopping.end_time = time.perf_counter()
        if best is None or evaluation < best or not shouldEvaluate and loss_now < loss_before:
            best = evaluation
            torch.save({
                "model":model.state_dict(),
                f"optimizer_{optimizer.__class__.__name__}" : optimizer.state_dict(),
                f"scheduler_{scheduler.__class__.__name__}" : scheduler.state_dict(),
                "epoch": epoch,
                "loss": loss.item() * gradient_accumulation_steps,
                "best" : best
            },model_weights_path + 'checkpoint.tar')
            # torch.save(model.state_dict(), model_weights_path+'model_weights.pth') # NOTE: Mozda puca
            # torch.save(optimizer.state_dict(), model_weights_path+f'optimizer_{optimizer.__class__.__name__}.pt') # NOTE: Mozda puca
            # torch.save(scheduler.state_dict(), model_weights_path+f'scheduler_{scheduler.__class__.__name__}.pt') # NOTE: Mozda puca

        if enable_early_stopping and (earlyStopping.step(evaluation if shouldEvaluate else loss.item() * gradient_accumulation_steps) or earlyStopping.time_ran_out()):
            break
    torch.backends.cudnn.benchmark = False
    progress_bar.close()
    return {"fid":fid_over_training, "losses":losses}