from data_provider.data_loader import Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    'ETTm1': Dataset_ETT_minute,
    'weather': Dataset_Custom,
}

dp_dict = {
    'ETTm1': 'ETT-small/ETTm1.csv',
    'weather': 'weather/weather.csv'
}

dl_dict = {
    'ETTm1': None,
    'weather': [ 'Tdew (degC)', 'T (degC)', 'Tpot (K)', 
                'VPmax (mbar)', 'rho (g/m**3)', 'VPdef (mbar)','Tlog (degC)']
}
def data_provider(args, flag):
    Data = data_dict[args.data]
    data_path = dp_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
    else:
        shuffle_flag = True

    data_set = Data(
        root_path="dataset",
        data_path=data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        labels = dl_dict[args.data]
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers)
    return data_set, data_loader
