import pytorch_lightning as pl
import torch
from typing import List, Tuple, Optional, Callable
from omegaconf import DictConfig
from market_oracle_lib.data.data_funcs import create_data_sets
from market_oracle_lib.parse_conf import get_data_getter_fn, get_data_symbols


class MyDataModule(pl.LightningDataModule):
    """A DataModule standardizes the training, val, test splits, data preparation and
    transforms. The main advantage is consistent data splits, data preparation and
    transforms across models.

    Example::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self, stage):
                # make assignments here (val/train/test split)
                # called on every process in DDP
            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)
            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)
            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)
            def teardown(self):
                # clean up after fit or test
                # called on every process in DDP
    """

    def __init__(self,
                 config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        data_getter_fn = get_data_getter_fn(self.config["data_loading"]["data_getter_fn_name"])
        symbols = get_data_symbols(self.config["data_loading"]["data_getter_fn_name"])
        print(f"{symbols=}")
        print(f"{self.config["data_loading"]["data_getter_fn_name"]=}")

        train_data, val_data, test_data = create_data_sets(
            data_getter_fn,
            symbols=symbols,
            **self.config["data_loading"]["data_getter_kwargs"]
        )
        self.train_data = train_data
        print(f"{self.train_data=}")
        print(f"{self.train_data.X.shape=}")
        self.val_data = val_data
        self.test_data = test_data

    def prepare_data(self):
        """Use this to download and prepare data. Downloading and saving data with
        multiple processes (distributed settings) will result in corrupted data.
        Lightning ensures this method is called only within a single process, so you can
        safely add your downloading logic within.

        Warning::
            DO NOT set state to the model (use ``setup`` instead)
            since this is NOT called on every device

        Example::

            def prepare_data(self):
                # good
                download_data()
                tokenize()
                etc()

                # bad
                self.split = data_split
                self.some_state = some_other_state()

        In a distributed environment, ``prepare_data`` can be called in two ways
        (using :ref:`prepare_data_per_node<common/lightning_module:prepare_data_per_node>`)

        1. Once per node. This is the default and is only called on LOCAL_RANK=0.
        2. Once in total. Only called on GLOBAL_RANK=0.

        Example::

            # DEFAULT
            # called once per node on LOCAL_RANK=0 of that node
            class LitDataModule(LightningDataModule):
                def __init__(self):
                    super().__init__()
                    self.prepare_data_per_node = True


            # call on GLOBAL_RANK=0 (great for shared file systems)
            class LitDataModule(LightningDataModule):
                def __init__(self):
                    super().__init__()
                    self.prepare_data_per_node = False

        This is called before requesting the dataloaders:

        .. code-block:: python

            dm.prepare_data()
            initialize_distributed()
            dm.setup(stage)
            dm.train_dataloader()
            dm.val_dataloader()
            dm.test_dataloader()
            dm.predict_dataloader()
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Called at the beginning of fit (train + validate), validate, test, or predict.
        This is a good hook when you need to build models dynamically or adjust something
        about them. This hook is called on every process when using DDP.

        setup is called from every process across all the nodes. Setting state here is
        recommended.

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``

        Example::

            class LitModel(...):
                def __init__(self):
                    self.l1 = None

                def prepare_data(self):
                    download_data()
                    tokenize()

                    # don't do this
                    self.something = else

                def setup(self, stage):
                    data = load_data(...)
                    self.l1 = nn.Linear(28, data.num_classes)
        """
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Implement one or more PyTorch DataLoaders for training.

        Return:
            A collection of :class:`torch.utils.data.DataLoader` specifying training samples.
            In the case of multiple dataloaders, please see this :ref:`section <multiple-dataloaders>`.

        The dataloader you return will not be reloaded unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_n_epochs` to
        a positive integer.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data

        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        Example::

            # single dataloader
            def train_dataloader(self):
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (1.0,))])
                dataset = MNIST(root='/path/to/mnist/', train=True, transform=transform,
                                download=True)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=True
                )
                return loader

            # multiple dataloaders, return as list
            def train_dataloader(self):
                mnist = MNIST(...)
                cifar = CIFAR(...)
                mnist_loader = torch.utils.data.DataLoader(
                    dataset=mnist, batch_size=self.batch_size, shuffle=True
                )
                cifar_loader = torch.utils.data.DataLoader(
                    dataset=cifar, batch_size=self.batch_size, shuffle=True
                )
                # each batch will be a list of tensors: [batch_mnist, batch_cifar]
                return [mnist_loader, cifar_loader]

            # multiple dataloader, return as dict
            def train_dataloader(self):
                mnist = MNIST(...)
                cifar = CIFAR(...)
                mnist_loader = torch.utils.data.DataLoader(
                    dataset=mnist, batch_size=self.batch_size, shuffle=True
                )
                cifar_loader = torch.utils.data.DataLoader(
                    dataset=cifar, batch_size=self.batch_size, shuffle=True
                )
                # each batch will be a dict of tensors: {'mnist': batch_mnist, 'cifar': batch_cifar}
                return {'mnist': mnist_loader, 'cifar': cifar_loader}
        """
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["training"]["num_workers"],
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Implement one or multiple PyTorch DataLoaders for validation.

        The dataloader you return will not be reloaded unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_n_epochs` to
        a positive integer.

        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`
        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.validate`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware
            There is no need to set it yourself.

        Return:
            A :class:`torch.utils.data.DataLoader` or a sequence of them specifying validation samples.

        Examples::

            def val_dataloader(self):
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (1.0,))])
                dataset = MNIST(root='/path/to/mnist/', train=False,
                                transform=transform, download=True)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=False
                )

                return loader

            # can also return multiple dataloaders
            def val_dataloader(self):
                return [loader_a, loader_b, ..., loader_n]

        Note:
            If you don't need a validation dataset and a :meth:`validation_step`, you don't need to
            implement this method.

        Note:
            In the case where you return multiple validation dataloaders, the :meth:`validation_step`
            will have an argument ``dataloader_idx`` which matches the order here.
        """
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["num_workers"],
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Implement one or multiple PyTorch DataLoaders for testing.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data


        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.test`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        Return:
            A :class:`torch.utils.data.DataLoader` or a sequence of them specifying testing samples.

        Example::

            def test_dataloader(self):
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (1.0,))])
                dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform,
                                download=True)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=False
                )

                return loader

            # can also return multiple dataloaders
            def test_dataloader(self):
                return [loader_a, loader_b, ..., loader_n]

        Note:
            If you don't need a test dataset and a :meth:`test_step`, you don't need to implement
            this method.

        Note:
            In the case where you return multiple test dataloaders, the :meth:`test_step`
            will have an argument ``dataloader_idx`` which matches the order here.
        """
        pass

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Implement one or multiple PyTorch DataLoaders for prediction.

        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware
            There is no need to set it yourself.

        Return:
            A :class:`torch.utils.data.DataLoader` or a sequence of them specifying prediction samples.

        Note:
            In the case where you return multiple prediction dataloaders, the :meth:`predict_step`
            will have an argument ``dataloader_idx`` which matches the order here.
        """
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["num_workers"],
        )

    def teardown(self, stage: str) -> None:
        """Called at the end of fit (train + validate), validate, test, or predict.

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """
        pass
