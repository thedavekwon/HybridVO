from model import *
from utils import *
from constants import *

device = torch.device(CUDA_DEVICE_NUM if torch.cuda.is_available() else "cpu")
weights = [1]
seqs = [["0" + str(i)] if i < 10 else [str(i)] for i in range(0, 11)]

for weight_num in weights:
    cur, model, optimizer, seq = load_model(
        HybridVO(),
        device,
        OPTIMIZER,
        0.001,
        path=f"{WEIGHT_FOLDER}/{weight_num}.weights",
        # path=weight_num
    )

    for seq in seqs:
        print(weight_num, seq[0])
        test_dl = DataLoader(
            KittiPredefinedDataset(seq, ortho6d=True),
            batch_size=DRAW_BATCH_SIZE,
            shuffle=False,
            num_workers=DRAW_NUM_WORKERS,
            drop_last=False,
        )
        test_rot6d(model, test_dl, device, weight_num, seq[0])
    del model, seq
    torch.cuda.empty_cache()