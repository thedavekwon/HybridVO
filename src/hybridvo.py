from model import *
from utils import *
from constants import *

device = torch.device(CUDA_DEVICE_NUM if torch.cuda.is_available() else "cpu")
cur, model, optimizer, scheduler = load_model(
    HybridVO(), device, OPTIMIZER, lr=0.001, path=""
)
if HALF:
    model.half()

train_dl = DataLoader(
    KittiPredefinedDataset(path=KITTI_PATH, ortho6d=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=True,
)

test_dl = DataLoader(
    KittiPredefinedDataset(seqs=[DRAW_SEQ], path=KITTI_PATH, ortho6d=True),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    drop_last=False,
)

epoch_losses = []

for e in range(cur + 1, cur + EPOCH + 1):
    epoch_loss = train(
        model, train_dl, optimizer, device, e, WEIGHT_FOLDER, LOSS_FOLDER
    )
    if e % 10 == 0 or (len(epoch_losses) == 0 or epoch_loss < min(epoch_losses)):
        print(f"{e}th epoch saved")
        model.save(e, optimizer)
        plt.clf()
        plt.plot(epoch_losses)
        plt.savefig(f"{LOSS_FOLDER}/train_{e}.png")
        test_rot6d(model, test_dl, device, e)
    epoch_losses.append(epoch_loss)
