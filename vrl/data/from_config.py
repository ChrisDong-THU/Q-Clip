import importlib
import lightning as L
from torch.utils.data import DataLoader


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    mod = importlib.import_module(module)
    if reload:
        importlib.reload(mod)
    return getattr(mod, cls)


def instantiate_from_target(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    cls_ = get_obj_from_str(config["target"])
    params = config["params"]
    return cls_(**params)


class DataModuleFromConfig(L.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=10, train=None, validation=None, test=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs["train"] = train
            
        if validation is not None:
            self.dataset_configs["validation"] = validation
            
        if test is not None:
            self.dataset_configs["test"] = test

    def setup(self, stage):
        self.datasets = dict(
            (k, instantiate_from_target(self.dataset_configs[k]))
            for k in self.dataset_configs)
        
    def train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size, num_workers=self.num_workers, 
                            shuffle=False, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.datasets["validation"], batch_size=self.batch_size, num_workers=self.num_workers, 
                            shuffle=False, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size, num_workers=self.num_workers, 
                            shuffle=False, pin_memory=True)
