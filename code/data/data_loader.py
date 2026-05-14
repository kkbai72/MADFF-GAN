
def CreateDataLoader(opt,is_val=False):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    if is_val:
        data_loader.initialize(opt,is_val)
    else:
        data_loader.initialize(opt)
    return data_loader
