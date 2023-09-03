from model import CICAFFModel 
from imports import np, torch
from helper_functions import load_data
from softmax_annealed_mean import SoftmaxAnnealedMean
probs = np.load('./prior_probs.npy', mmap_mode='r')
mappings = np.load('./pts_in_hull.npy', mmap_mode='r')
num_classes = len(probs)
ab_range = np.arange(-110, 120, 10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CICAFFModel(number_of_classes=num_classes).to(device)
data_memmap = load_data('r')
data_sample = torch.as_tensor(data_memmap[np.random.randint(0,len(data_memmap))].copy(),dtype=torch.float, device=device)

X, Y = data_sample[0:1], data_sample[1:]

print(X.shape)

# model_outputs = model(X)
# NOTE: outputi
encoder1_output = model.en1(X.reshape(1,1,224,224))
encoder2_output = model.en2(encoder1_output)
encoder3_output = model.en3(encoder2_output)
encoder4_output = model.en4(encoder3_output)
encoder5_output = model.en5(encoder4_output)
encoder6_output = model.en6(encoder5_output)
classification_subnet_output = model.classification_subnet(encoder6_output)
# NOTE : printovanje
print("encoder1_output has output of shape:\t" , encoder1_output.shape)
print("encoder2_output has output of shape:\t" , encoder2_output.shape)
print("encoder3_output has output of shape:\t" , encoder3_output.shape)
print("encoder4_output has output of shape:\t" , encoder4_output.shape)
print("encoder5_output has output of shape:\t" , encoder5_output.shape)
print("encoder6_output has output of shape:\t" , encoder6_output.shape)
print("classification_subnet_output has output of shape:\t" , classification_subnet_output.shape)
print("classification_subnet_output sums:\t",classification_subnet_output.sum(dim=1))
#NOTE: Outputi modela
# assert model_output.shape[1] == num_classes, f"Ocekivana dubina na izlazu {num_classes}, dobijena: {model_output.shape[1]}"
# random_classes = np.random.randint(0,num_classes,15)
# for class_num in random_classes:
#     print(f"Klasa broj {class_num} ima vrednosti:\n{model_output[0][class_num][0:5][0:5]}\n")

# softmax = SoftmaxAnnealedMean()

# softmax_annealed_output = softmax(model_output)
# print(softmax_annealed_output.shape)
# print(softmax_annealed_output.sum(dim=1).shape)