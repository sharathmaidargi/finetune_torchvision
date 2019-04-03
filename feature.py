import torchvision.models
import sklearn.svm
import torch.nn as nn
import numpy as np
import torch.utils.data
import  matplotlib.pyplot as plt
import torchvision.transforms as transforms
from inception import inception_v3
from util import plot_confusion_matrix

model = inception_v3(pretrained=True)
model.aux_logit=False
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
resize = transforms.Resize((299, 299))

preprocessor = transforms.Compose([
    resize,
    transforms.ToTensor(),
    normalize,
])


in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 9)

# new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
# model.classifier = new_classifier

data_dir1 = "/home/sharathmaidargi/Desktop/DeepLearning/PreProject/src/datasets/household/train"
batch_size1 = 32

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(data_dir1, preprocessor),
    batch_size=batch_size1,
    shuffle=True)

data_dir2 = "/home/sharathmaidargi/Desktop/DeepLearning/PreProject/src/datasets/household/test"
batch_size2 = 32
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(data_dir2, preprocessor),
    batch_size=batch_size2,
    shuffle=True)
x_train = None
y_train = None
x_test = None
y_test = None

for i, (in_data, target) in enumerate(train_loader):
    print i
    input_var = torch.autograd.Variable(in_data)
    target_var = torch.autograd.Variable(target)
    output = np.asarray( model(input_var))
    # print output
    # convert the output of feature extractor to numpy array

    if x_train is None:
        x_train = output
    else:
        x_train = np.append(x_train, output,axis=0)

    if y_train is None:
        y_train = target_var.data.numpy()
    else:
        y_train = np.append(y_train, target_var.data.numpy(), axis=0)


for i, (in_data, target) in enumerate(test_loader):
    input_var = torch.autograd.Variable(in_data)
    target_var = torch.autograd.Variable(target)
    output = np.asarray(model(input_var))
    # convert the output of feature extractor to numpy array

    if x_test is None:
        x_test = output #output.data.numpy()
    else:
        x_test = np.append(x_test, output, axis=0)

    if y_test is None:
        y_test = target_var.data.numpy()
    else:
        y_test = np.append(y_test, target_var.data.numpy(), axis=0)


gaussian_model = sklearn.svm.SVC(C=1.0, kernel='poly', degree=6, coef0=4)

loss = gaussian_model.fit(x_train, y_train)
y_pred = gaussian_model.predict(x_test)
accuracy = gaussian_model.score(x_test, y_test)

print "Scikit SVM for Inception feature extractor Dataset : " + str(accuracy.round(4)) + "\n\n"
print "Algorithm|Accuracy\n"

plot_confusion_matrix(y_pred, y_test)
plt.show()




