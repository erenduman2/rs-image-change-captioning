from helpers import get_eval_score


def test(ds, processor, model, device):

    references = list()
    hypotheses = list()

    model.eval()
    model.to(device)

    # for i in range(10): # ilk 10 veri için skor üretme
    for i in range(len(ds)):
        image = ds[i]["image"] # image oku
        input = processor(images=image, return_tensors="pt").to(device) # image process et
        pixel_values = input.pixel_values # pixel değerlerini al
        generated_ids = model.generate(pixel_values=pixel_values, max_length=20)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0] # decode ile cümle üret
        print("image: ", i, "  ", generated_caption)
        hypotheses.append(generated_caption)
        print(ds[i]["captions"])
        references.append(ds[i]["captions"])

    hypotheses = [[item] for item in hypotheses]

    metrics = get_eval_score(references, hypotheses)

    print("BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} ROUGE_L {} CIDEr {}, Meteor {}".format
          (metrics["Bleu_1"], metrics["Bleu_2"], metrics["Bleu_3"],
           metrics["Bleu_4"], metrics["ROUGE_L"], metrics["CIDEr"], metrics["METEOR"]))


    return metrics

