from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# region image

def create_image_dataloaders(
    dataset_type: str,
    dataset_name: str,
    batch_size: int,
    num_workers: int
):
    transform =  get_transform_fn(dataset_name)

    train_dataset, test_dataset, dataset_config = create_image_dataset(dataset_type, dataset_name, transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, dataset_config


def create_image_dataset(
    dataset_type: str,
    dataset_name: str,
    transform: transforms.Compose
):
    if dataset_type == "manual":
        return get_manual_image_dataset(dataset_name, transform)
    elif dataset_type == "pre-built":
       return get_pre_built_image_dataset(dataset_name, transform)
    else:
       raise ValueError(f"Dataset type is unrecgognised! Got {dataset_type}")


def get_pre_built_image_dataset(
    dataset_name: str,
    transform: transforms.Compose
):
    dataset_names = {
        "mnist": {
            "train": lambda: datasets.MNIST(root="data", train=True, download=True, transform=transform),
            "test": lambda: datasets.MNIST(root="data", train=False, download=True, transform=transform),
            "config":  { "channel": 1, "height": 28, "width": 28, "patch_size": 7 }
        },
        "cifar-10": {
            "train": lambda: datasets.CIFAR10(root="data", train=True, download=True, transform=transform),
            "test": lambda: datasets.CIFAR10(root="data", train=False, download=True, transform=transform),
            "config":  { "channel": 3, "height": 32, "width": 32, "patch_size": 8 }
        }
    }

    if dataset_name not in dataset_names:
        raise ValueError(
            f"Dataset type is unrecgognised! Got {dataset_name}. Available datasets are: {list(dataset_names.keys())}"
        )

    train_dataset = dataset_names[dataset_name]["train"]()
    test_dataset = dataset_names[dataset_name]["test"]()
    dataset_config = dataset_names[dataset_name]["config"]

    dataset_config["n_classes"] = len(train_dataset.classes)

    return train_dataset, test_dataset, dataset_config


def get_manual_image_dataset(
    dataset_name: str,
    transform: transforms.Compose
):
    train_dataset = datasets.ImageFolder(f"{dataset_name}/train", transform=transform)
    test_dataset = datasets.ImageFolder(f"{dataset_name}/test", transform=transform)
    dataset_config = {}

    dataset_config["n_classes"] = len(train_dataset.classes)

    return train_dataset, test_dataset, dataset_config


def get_transform_fn(
    dataset_type: str
) -> transforms.Compose:

    if dataset_type == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_type == "cifar-10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
       raise ValueError(f"Dataset type is unrecgognised for transform fn! Got {dataset_type}")

    return transform

# endregion
