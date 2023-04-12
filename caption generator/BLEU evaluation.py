from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import numpy as np
from Inference import *

# greedy
def compute_bleu_greedy(filename, captions_dict, model):
    ref = []
    for item in captions_dict[filename]:
        ref.append(item.split()[1:-1])
    fv = np.array([img_feature_vectors[filename]])
    cand = decode_caption(generate_caption(fv, model, vocab), vocab).split()[1:-1]
    return np.array((sentence_bleu(ref, cand, weights=(1.0, 0, 0, 0), smoothing_function=SmoothingFunction().method1), 
    sentence_bleu(ref, cand, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method1),
    sentence_bleu(ref, cand, weights=(0.3, 0.3, 0.3, 0), smoothing_function=SmoothingFunction().method1),
    sentence_bleu(ref, cand, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)))

# beam search
def compute_bleu_beam(filename, captions_dict, model):
    ref = []
    for item in captions_dict[filename]:
        ref.append(item.split()[1:-1])
    fv = np.array([img_feature_vectors[filename]])
    complete_captions = beam_search(fv, model, 5, 5, vocab)
    sorted_list = sorted(complete_captions, key=lambda x: x[1])
    cand = decode_caption(sorted_list[-1][0], vocab).split()[1:-1]
    return np.array((sentence_bleu(ref, cand, weights=(1.0, 0, 0, 0), smoothing_function=SmoothingFunction().method1), 
    sentence_bleu(ref, cand, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method1),
    sentence_bleu(ref, cand, weights=(0.3, 0.3, 0.3, 0), smoothing_function=SmoothingFunction().method1),
    sentence_bleu(ref, cand, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)))

# nucleus
def compute_bleu_ns(filename, captions_dict, model):
    ref = []
    for item in captions_dict[filename]:
        ref.append(item.split()[1:-1])
    fv = np.array([img_feature_vectors[filename]])
    cand = decode_caption(generate_caption_nucleus_sampling(fv, model, vocab), vocab).split()[1:-1]
    return np.array((sentence_bleu(ref, cand, weights=(1.0, 0, 0, 0), smoothing_function=SmoothingFunction().method1), 
    sentence_bleu(ref, cand, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method1),
    sentence_bleu(ref, cand, weights=(0.3, 0.3, 0.3, 0), smoothing_function=SmoothingFunction().method1),
    sentence_bleu(ref, cand, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)))

def compute_avg_bleu(alg):
    if alg == "greedy":
        bleu_func = compute_bleu_greedy
    elif alg == "beam":
        bleu_func = compute_bleu_beam
    elif alg == "nucleus_sampling":
        bleu_func = compute_bleu_ns
    else:
        return 0
    scores = {"BLEU-1":[], "BLEU-2":[], "BLEU-3":[], "BLEU-4":[]}
    n=0
    for item in filenames_test:
        result = bleu_func(item, captions_dict, model)
        scores["BLEU-1"].append(result[0])
        scores["BLEU-2"].append(result[1])
        scores["BLEU-3"].append(result[2])
        scores["BLEU-4"].append(result[3])
        print(str(n) + "/" + str(len(filenames_test)))
        n+=1

    print(np.average(scores['BLEU-1']))
    print(np.average(scores['BLEU-2']))
    print(np.average(scores['BLEU-3']))
    print(np.average(scores['BLEU-4']))

if __name__ == "__main__":
    max_length = 40
    vocab_size = 5185
    model = create_model(max_length, vocab_size)
    model.load_weights("data/final_model_weights.h5")
    vocab = np.load("data/vocab.npy", allow_pickle=True).item()
    filenames_test = np.load("data/filenames_test.npy")
    captions_dict = np.load("data/captions_dict.npy", allow_pickle=True).item()
    img_feature_vectors = np.load("data/img_feature_vectors.npy", allow_pickle=True).item()
    # argumenets: greedy, beam or nucleus_sampling
    compute_avg_bleu("greedy")