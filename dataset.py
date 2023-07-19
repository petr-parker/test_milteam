from torchvision import transforms
from torchvision.datasets import MNIST


def get_dataset():
    MEAN_MNIST = [0.1307]
    STD_MNIST = [0.3081]

    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN_MNIST, STD_MNIST)
    ]

    train_transform = transforms.Compose(normalize)
    valid_transform = transforms.Compose(normalize)

    dataset_train = MNIST(root=".\data", train=True, download=True, transform=train_transform)
    dataset_valid = MNIST(root=".\data", train=False, download=True, transform=valid_transform)

    return dataset_train, dataset_valid
