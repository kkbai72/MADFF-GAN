import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt,is_val=False):
    dataset = None
    if opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'unaligned_random_crop':
        from data.unaligned_random_crop import UnalignedDataset
        dataset = UnalignedDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    if is_val:
        dataset.initialize(opt,is_val)
    else:
        dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt,is_val=False):
        BaseDataLoader.initialize(self, opt)
        if is_val:
            self.dataset = CreateDataset(opt,is_val)
        else:
            self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
