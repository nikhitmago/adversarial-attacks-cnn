"""
Adversary Attack example code
"""
# from torchvision import models as tvm
import torch
from torch.nn import functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AdversialAttacker(object):
    def __init__(self, method='FGSM'):
        assert method in ['FGSM', 'I-FGSM']
        self.method = method
        self.criterion = nn.CrossEntropyLoss()
        #print("created adversial attacker in method '%s'" % (method))

    def get_pred_label(self, mdl, inp, ret_out_scores=False, ret_out_pred=True):
        # use current model to get predicted label
        train = mdl.training
        mdl.eval()
        with torch.no_grad():
            out = F.softmax(mdl(inp), dim=1)
        out_score, out_pred = out.max(dim=1)
        if ret_out_scores and not ret_out_pred:
            return out
        if ret_out_pred and not ret_out_scores:
            return out_pred
        mdl.train(train)
        return out_pred, out

    def perturb_untargeted(self, mdl, input_tensor, targ_label=None, eps=0.3, num_steps=10, alpha=0.025):
        # perform attacking perturbation in the untargeted setting
        # note: feel free the change the function arguments for your implementation
        mdl.train()
        inp = input_tensor.detach().clone()
        inp = inp.requires_grad_(True)
        if self.method == 'FGSM':
            outputs = mdl(inp)
            mdl.zero_grad()

            loss = self.criterion(outputs, targ_label)
            loss.backward()

            x_grad = inp.grad.detach()
            x_grad_sign = x_grad.sign()

            inp_adv = inp + eps * x_grad_sign
            inp_adv = torch.clamp(inp_adv, 0, 1)

            mdl.eval()
            return inp_adv
        
        elif self.method == 'I-FGSM':
            for i in range(num_steps):
                outputs = mdl(inp)
                mdl.zero_grad()

                loss = self.criterion(outputs, targ_label)
                loss.backward()

                x_grad = inp.grad.detach()
                x_grad_sign = x_grad.sign()

                temp_adv = inp + alpha * x_grad_sign
                total_grad = temp_adv - input_tensor
                total_grad = torch.clamp(total_grad, -eps, eps)

                x_adv = input_tensor + total_grad
                inp.data = x_adv
            
            mdl.eval()
            return inp

    def perturb_targeted(self, mdl, input_tensor, targ_label=None, eps=0.3, num_steps=10, alpha=0.075):
        # perform attacking perturbation in the targeted setting
        # note: feel free the change the function arguments for your implementation
        mdl.train()
        inp = input_tensor.detach().clone()
        inp = inp.requires_grad_(True)
        if self.method == 'FGSM':
            outputs = mdl(inp)
            mdl.zero_grad()

            loss = self.criterion(outputs, targ_label)
            loss.backward()

            x_grad = inp.grad.detach()
            x_grad_sign = x_grad.sign()

            inp_adv = inp - eps * x_grad_sign
            inp_adv = torch.clamp(inp_adv, 0, 1)

            mdl.eval()
            return inp_adv
        
        elif self.method == 'I-FGSM':
            for i in range(num_steps):
                outputs = mdl(inp)
                mdl.zero_grad()

                loss = self.criterion(outputs, targ_label)
                loss.backward()

                x_grad = inp.grad.detach()
                x_grad_sign = x_grad.sign()

                temp_adv = inp - alpha * x_grad_sign
                total_grad = temp_adv - input_tensor
                total_grad = torch.clamp(total_grad, -eps, eps)

                x_adv = input_tensor + total_grad
                inp.data = x_adv
            
            mdl.eval()
            return inp


class Clamp:
    def __call__(self, inp):
        return torch.clamp(inp, 0., 1.)


def generate_experiment(method='FGSM', img_name='CINIC-10/test/airplane/cifar10-test-10.png') :

    # define your model and load pretrained weights
    model = Net()
    model.load_state_dict(torch.load('best_model.pth.tar', map_location=torch.device('cpu')))
    
    # cinic class names
    import yaml
    with open('./cinic_classnames.yml', 'r') as fp:
        classnames = yaml.safe_load(fp)

    # load image
    img_path = Path(img_name)
    input_img = Image.open(img_path).convert('RGB')

    # define normalizer and un-normalizer for images
    # cinic
    mean = [0.47889522, 0.47227842, 0.43047404]
    std = [0.24205776, 0.23828046, 0.25874835]

    tf_img = transforms.Compose(
        [
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std
            )
        ]
    )
    un_norm = transforms.Compose(
        [
            transforms.Normalize(
                mean=[-m/s for m, s in zip(mean, std)],
                std=[1/s for s in std]
            ),
            Clamp(),
            transforms.ToPILImage()
        ]
    )

    # To be used for iterative method
    # to ensure staying within Linf limits
    clip_min = min([-m/s for m, s in zip(mean, std)])
    clip_max = max([(1-m)/s for m, s in zip(mean, std)])

    input_tensor = tf_img(input_img)
    attacker = AdversialAttacker(method=method)

    return {
        'img': input_img,
        'inp': input_tensor.unsqueeze(0),
        'attacker': attacker,
        'mdl': model,
        'clip_min': clip_min,
        'clip_max': clip_max,
        'un_norm': un_norm,
        'classnames': classnames
    }
