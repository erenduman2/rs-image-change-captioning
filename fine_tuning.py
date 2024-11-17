import train
from transformers import AutoProcessor
from transformers import BlipForConditionalGeneration
import torch

from dataset import rsicd_data_loader, levir_cc_dataloader
from test import test
from train import train
from helpers import save_epoch_metrics, find_latest_model


def fine_tune():

    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("device:", device)


    #     Image Captioning Dataloader
    train_dataloader = rsicd_data_loader("train", processor)
    valid_dataloader = rsicd_data_loader("valid", processor)
    test_dataset = rsicd_data_loader("test")

    #     Change Captioning Dataloader
    # train_dataloader = levir_cc_dataloader("train", processor)
    # valid_dataloader = levir_cc_dataloader("valid", processor)
    # test_dataset = load_dataset("levir_cc", split="test")  # load train dataset
    # test_dataset = concatenation/subtraction/hadamard(test_dataset)


    #   Find from which epoch to start to train.
    latest_model = find_latest_model(folder_path="../saved_models")
    if latest_model is not None:
        checkpoint = torch.load('../saved_models/{}'.format(latest_model), weights_only=False)  # This model path should be CHANGED, it needs to be DYNAMIC
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1  # Start from new epoch
        print(f"Resuming training from epoch {starting_epoch}...")
    else:
        print("No checkpoint found. Starting to train from scratch.")
        starting_epoch = 0  # Start from scratch


    for epoch in range(starting_epoch, 10):
        print("Epoch:", epoch)

        avg_train_loss, avg_valid_loss, model, optimizer = train(train_dataloader, valid_dataloader, model, optimizer, device, epoch)

        metrics = test(test_dataset, processor, model, device)

        save_epoch_metrics(epoch, avg_train_loss, avg_valid_loss, metrics["Bleu_1"],
                           metrics["Bleu_2"], metrics["Bleu_3"], metrics["Bleu_4"],
                           metrics["ROUGE_L"], metrics["CIDEr"], metrics["METEOR"])


fine_tune()