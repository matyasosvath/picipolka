from typing import Optional

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# region image

def create_image_dataloaders(
    dataset_type: str,
    dataset_name: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int
):

  train_dataset, test_dataset, class_names = create_image_dataset(dataset_type, dataset_name, transform)

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

  return train_dataloader, test_dataloader, class_names


def create_image_dataset(
    dataset_type: str,
    dataset_name: str,
    transform: Optional[transforms.Compose]
):

    if dataset_type == "manual":
        assert transform is not None, "transform function must be assigned for manual dataset"
        return get_manual_image_dataset(dataset_name, transform)

    elif dataset_type == "pre-built":
       return get_pre_built_image_dataset(dataset_name, transform)
    else:
       raise ValueError(f"Dataset type is unrecgognised! Got {dataset_type}")


def get_pre_built_image_dataset(
    dataset_name: str,
    transform: Optional[transforms.Compose]
):
    transform_fn =  transform or get_transform_fn(dataset_name)

    dataset_names = {
        "mnist": {
            "train": lambda: datasets.MNIST(root="data", train=True, download=True, transform=transform_fn),
            "test": lambda: datasets.MNIST(root="data", train=False, download=True, transform=transform_fn),
            "config":  { "channel": 1, "height": 28, "width": 28, "patch_size": 7 }
        },
        "cifar-10": {
            "train": lambda: datasets.CIFAR10(root="data", train=True, download=True, transform=transform_fn),
            "test": lambda: datasets.CIFAR10(root="data", train=False, download=True, transform=transform_fn),
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


def get_manual_image_dataset(dataset_name: str, transform: transforms.Compose):

    train_dataset = datasets.ImageFolder(f"{dataset_name}/train", transform=transform)
    test_dataset = datasets.ImageFolder(f"{dataset_name}/test", transform=transform)

    class_names = train_dataset.classes

    return train_dataset, test_dataset, class_names


def get_transform_fn(dataset_type: str) -> transforms.Compose:

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
