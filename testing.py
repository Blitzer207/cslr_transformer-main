"""
This script tests the model
"""
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.aslr import Transformer as T
from utils.wer import compute_wer, seq_to_text, seq_to_text_reverse, compute_wer_complete
from torchsummary import summary
from utils.datasets import VideoDataset
from utils.error_analysis import substitution_error
import argparse
parser = argparse.ArgumentParser(description="Arguments for the testing script")
parser.add_argument(
    "--multisigner",
    type=bool,
    help="Boolean to specify if we use multisigner data or signer independent data.",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Integer to specify the batch_size",
)
parser.add_argument(
    "--path_to_model",
    type=str,
    default="transformer.pt",
    help="String to specify to the nodel to test"
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="String to specify device to train on"
)
args = parser.parse_args()
TEST_BATCH_SIZE = 1
if __name__ == "__main__":
    to_use = "multi" if args.multisigner else "si5"
    trg_vocab = 1297 if args.multisigner else 1137
    PADDING = 1296 if args.multisigner else 1136
    ground_truth, prediction = [], []
    vocab = np.load("full_gloss_" + to_use + ".npy", allow_pickle=True)
    vocab = dict(enumerate(vocab.flatten()))
    vocab = vocab[0]
    soft = torch.nn.Softmax(dim=1)
    print("Following device will be used: ", args.device)
    print("The model in the folder " + args.path_to_model + " will be tested with a batch size of ", args.batch_size)
    model = T(
        device=torch.device(str(args.device).lower()),
        embed_dim=512,
        seq_length_en=300,
        seq_length_dec=300,
        src_vocab_size=300,
        target_vocab_size=trg_vocab,
        num_layers=8,
        expansion_factor=4,
        n_heads=8,
        dropout_rate=0.2,
    )
    if torch.cuda.device_count() > 1 and args.device.lower() == "cuda":
        # You may need to adapt this to your GPU environment
        dev_arr = np.arange(torch.cuda.device_count())
        visible_device=""
        for dev in dev_arr:
            visible_device = visible_device + str(dev) + ","
        visible_device = visible_device[:-1]
        # os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
        print("We will be training on {0} GPUs with IDs" + visible_device)
        model = torch.nn.DataParallel(model, device_ids=[0,1])
    model.load_state_dict(torch.load(args.path_to_model))
    model.to(torch.device(str(args.device).lower()))
    summary(model=model)
    test_features = np.load(to_use + "_testing_features.npy", allow_pickle=True)
    test_labels = np.load(to_use + "_testing_label.npy", allow_pickle=True)
    test_d = VideoDataset(feature_tab=test_features, label=test_labels)
    test_set = DataLoader(
            dataset=test_d,
            batch_size=TEST_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )
    model.eval()
    file = open(to_use + "testing_results.txt", "w")
    file.write("Path to model being tested : " + args.path_to_model)
    file.write("\n\n")
    with torch.no_grad():
        for src, trg in tqdm(test_set, desc="Testing"):
            src, trg = src.to(torch.device(str(args.device).lower())), trg.to(torch.device(str(args.device).lower()))
            predictions = model(src, trg)
            for i in range(trg.shape[0]):
                i_trg = trg[i]
                i_pred = predictions[i]
                i_pred = soft(i_pred)
                i_pred = torch.argmax(i_pred, dim=-1)
                gt_s, pre_s = seq_to_text(
                    vocab=vocab, padding_character=PADDING, ground_truth=i_trg, prediction=i_pred
                )
                prediction.append(pre_s)
                ground_truth.append(gt_s)
                file.write("\n")
                file.write("Ground Truth sentence : " + ground_truth[-1])
                file.write("\n")
                file.write("Predicted sentence : " + prediction[-1])
                file.write("\n\n")
    results = compute_wer_complete(ground_truth=ground_truth, prediction=prediction)
    wer = results["wer"] * 100
    num_ins = results["insertions"]
    num_del = results["deletions"]
    num_sub = results["substitutions"]
    file.write("\n")
    file.write("Final Word Error Rate in % : " + str(round(wer, 2)) + "%")
    file.write("\n")
    file.write("Final number of insertions : " + str(num_ins))
    file.write("\n")
    file.write("Final number of substitutions : " + str(num_sub))
    file.write("\n")
    file.write("Final number of deletions % : " + str(num_del))
    file.write("\n")
    file.write("\n")
    correct, err, signer = substitution_error(ground_truth=ground_truth, prediction=prediction, multi=args.multisigner)
    dict_good = {"good": np.array(correct, dtype=str)}
    dict_bad = {"bad": np.array(err, dtype=str)}
    file.close()
    print("Final Word Error Rate in % : " + str(round(wer, 2)) + "%")
    print("Final number of insertions : " + str(num_ins))
    print("Final number of substitutions : " + str(num_sub))
    print("Final number of deletions % : " + str(num_del))
    print("error analysis error", err)
    print("error analysis correct", correct)
    np.save(to_use + "_error_analysis_error", np.array(err))
    np.save(to_use + "_error_analysis_correct", np.array(correct))
    np.save(to_use + "_error_analysis_signer", np.array(signer))
    print("Analysis saved as numpy array")




