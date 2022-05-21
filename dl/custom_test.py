from matplotlib import pyplot as plt
from .webcam_classifier import video_capture
import ipdb
import torch
import matplotlib.pyplot as plt
import matplotlib
from torchvision import transforms, utils

try:
    matplotlib.use("TkAgg")
except:
    pass


def custom_test(model):
    """
    Custom test function for the model.
    """
    print("Custom test function for the model.")

    video_capture(model)
    # dream_label(model)

    return


val_trans = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]
)


def dream_label(model):

    # impath = "./images/0.jpg"
    # # read image and convert to tensor
    # im = plt.imread(impath)
    # x = val_trans(im)

    # # ipdb.set_trace()
    # # x = x.permute(1,2,0)
    # x = x.unsqueeze(0)
    # model 
    # ipdb.set_trace()
    model.eval()
    x = torch.nn.Parameter(torch.ones(1, 1, 48, 48), requires_grad=True)
    optim = torch.optim.SGD([x], lr=1e-1)
    mse = torch.nn.CrossEntropyLoss()

    y = torch.tensor([1])  # the desired network response

    # disable parameter require grad of model to false
    for param in model.parameters():
        param.require_grad = False

    num_steps = 1000  # how many optim steps to take
    for _ in range(num_steps):
        loss = mse(model(x), y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        print("loss", loss.item())
        # print(model(x))
        # plot x
    img = x[0, 0, ...].detach().numpy() * 0.5 + 0.5
    plt.imshow(img, cmap="gray")
    plt.savefig("b.jpg")

    x.requires_grad = False
    # ipdb.set_trace()
    model.showActivations(x)
    
    # plt.show()
