import os
import re

import csv
from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor

def get_eval_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(references, hypotheses)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict

def find_latest_model(folder_path):
    files = os.listdir(folder_path)
    model_files = [f for f in files if re.match(r'model_epoch_\d+', f)]

    if not model_files:
        return None

    model_numbers = [int(re.search(r'\d+', f).group()) for f in model_files]
    highest_number = max(model_numbers)

    latest_model = "model_epoch_{}.pth".format(highest_number)
    print(latest_model)
    return latest_model

def save_batch_metrics(epoch_num, batch_num, current_training_loss, avg_training_loss):
    with open("../metrics/batch_metrics.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Create file with headers if not created before.
            writer.writerow(["Epoch Number", "Batch Number", "Current Training Loss", "Average Training Loss"])
        writer.writerow([epoch_num, batch_num, current_training_loss, avg_training_loss])

def save_epoch_metrics(epoch_num, training_loss, validation_loss, bleu_1, bleu_2, bleu_3, bleu_4, rouge_l, cider, meteor):
    with open("../metrics/epoch_metrics.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Create file with headers if not created before.
            writer.writerow(["Epoch Number", "Training Loss", "Validation Loss", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE_L", "CIDEr", "Meteor"])
        writer.writerow([epoch_num, training_loss, validation_loss, bleu_1, bleu_2, bleu_3, bleu_4, rouge_l, cider, meteor])