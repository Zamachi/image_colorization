from model import ColorizerModel
from imports import np, torch
from helper_functions import load_data
from softmax_annealed_mean import SoftmaxAnnealedMean
probs = np.load('./prior_probs.npy', mmap_mode='r')
mappings = np.load('./pts_in_hull.npy', mmap_mode='r')
num_classes = len(probs)
ab_range = np.arange(-110, 120, 10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ColorizerModel(number_of_classes=num_classes).to(device)
data_memmap = load_data('r')
data_sample = torch.as_tensor(data_memmap[np.random.randint(0,len(data_memmap))].copy(),dtype=torch.float, device=device)

X, Y = data_sample[0:1], data_sample[1:]

print(X.shape)

# model_outputs = model(X)
# NOTE: outputi
conv1_output = torch.nn.ReLU()(model.conv1(X.reshape(1,1,224,224)))
conv2_output = torch.nn.ReLU()(model.conv2(conv1_output))
conv3_output = torch.nn.ReLU()(model.conv3(conv2_output))
conv4_output = torch.nn.ReLU()(model.conv4(conv3_output))
conv5_output = torch.nn.ReLU()(model.conv5(conv4_output))
conv6_output = torch.nn.ReLU()(model.conv6(conv5_output))
conv7_output = torch.nn.ReLU()(model.conv7(conv6_output))
conv8_output = torch.nn.ReLU()(model.conv8(conv7_output))
conv9_output = torch.nn.ReLU()(model.conv9(conv8_output))
upsample1_output = model.upsample1(conv9_output)
conv10_output = torch.nn.ReLU()(model.conv10(upsample1_output))
upsample2_output = model.upsample2(conv10_output)
conv11_output = torch.nn.ReLU()(model.conv11(upsample2_output))
upsample3_output = model.upsample3(conv11_output)
model_output = model.model_output(upsample3_output)
# NOTE : printovanje
print("conv1_output has output of shape:\t" , conv1_output.shape)
print("conv2_output has output of shape:\t" , conv2_output.shape)
print("conv3_output has output of shape:\t" , conv3_output.shape)
print("conv4_output has output of shape:\t" , conv4_output.shape)
print("conv5_output has output of shape:\t" , conv5_output.shape)
print("conv6_output has output of shape:\t" , conv6_output.shape)
print("conv7_output has output of shape:\t" , conv7_output.shape)
print("conv8_output has output of shape:\t" , conv8_output.shape)
print("conv9_output has output of shape:\t" , conv9_output.shape)
print("upsample1_output has output of shape:\t" , upsample1_output.shape)
print("conv10_output has output of shape:\t" , conv10_output.shape)
print("upsample2_output has output of shape:\t" , upsample2_output.shape)
print("conv11_output has output of shape:\t" , conv11_output.shape)
print("upsample3_output has output of shape:\t" , upsample3_output.shape)
print("model_output has output of shape:\t" , model_output.shape)

#NOTE: Outputi modela
assert model_output.shape[1] == num_classes, f"Ocekivana dubina na izlazu {num_classes}, dobijena: {model_output.shape[1]}"
random_classes = np.random.randint(0,num_classes,15)
for class_num in random_classes:
    print(f"Klasa broj {class_num} ima vrednosti:\n{model_output[0][class_num][0:5][0:5]}\n")

softmax = SoftmaxAnnealedMean()

softmax_annealed_output = softmax(model_output)
print(softmax_annealed_output.shape)
print(softmax_annealed_output.sum(dim=1).shape)