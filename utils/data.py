from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10


# def _prepare_cifar10_exp():
#     input_shape = (3, 32, 32)
#     preprocess = [Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     dataset = get_cifar10_dataset(ds='test', normalize=False)
#     model = cifar10(pretrained=True,
#                     map_location='cpu' if not torch.cuda.is_available() else None)
#
#     return input_shape, dataset, preprocess, model

def get_cifar10_dataset(ds='train', normalize=True, download=True):
    transform_list = [ToTensor()]
    if normalize:
        transform_list.append(
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    transform = Compose(transform_list)

    dataset = CIFAR10(root='datasets/Cifar10', train=True if (ds == 'train') else False,
                      download=download, transform=transform)
    return dataset



if __name__ == '__main__':
    dataset = get_cifar10_dataset()
    print("")
