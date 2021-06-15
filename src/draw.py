from model import *
from utils import *
from constants import *

device = torch.device(CUDA_DEVICE_NUM if torch.cuda.is_available() else "cpu")
weights = list(range(10, 130, 5))
seqs = [["0" + str(i)] if i < 10 else [str(i)] for i in range(1, 11)]

for weight_num in weights:
    cur, model, optimizer, seq = load_model(DeepVO(), device, OPTIMIZER, 0.001, path=f"{WEIGHT_FOLDER}/{weight_num}.weights")
    for seq in seqs:
        test_dl = DataLoader(KittiPredefinedDataset(seq), batch_size=DRAW_BATCH_SIZE, shuffle=False, num_workers=DRAW_NUM_WORKERS)
        gt, pred, traj = test(model, test_dl, device, test_seq=seq)
        draw_route(gt, pred, f"{seq[0]}_test_{weight_num}")
        np.savetxt(f"{RESULT_FOLDER}/{seq}_{weight_num}_pred.txt", traj, fmt="%1.8f")
    del model, seq
    torch.cuda.empty_cache()