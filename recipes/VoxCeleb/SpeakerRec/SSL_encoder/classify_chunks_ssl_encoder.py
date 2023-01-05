#!/usr/bin/python3
"""Recipe for training a speaker verification system based on cosine distance.
The cosine distance is computed on the top of pre-trained embeddings.
The pre-trained model is automatically downloaded from the web if not specified.
This recipe is designed to work on a single GPU.

To run this recipe, run the following command:
    >  python speaker_verification_cosine.py hyperparams/verification_ecapa_tdnn.yaml

Authors
    * Hwidong Na 2020
    * Mirco Ravanelli 2020
"""
from audioop import reverse
import os
import sys
from turtle import pos
import torch
import logging
import torchaudio
import speechbrain as sb
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import EER, minDCF
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
from speechbrain.nnet.losses import classification_error
import numpy as np


def sample_chunks(wavs,lens):
    
    chunks = []
    indices = []
    actual_lens = lens * wavs.size()[1]

    padded_lens = actual_lens.clone()

    chunk_len = int(params["chunk_size"] * params["sample_rate"] / 1000)
    lens = lens.detach().cpu().numpy()

    for i in range(wavs.size()[0]):
        
        tmp_len = int(wavs[i].size()[0])
        if tmp_len >= chunk_len:
        
            chunk_anchors = torch.randint(chunk_len,tmp_len,(2,1))
        
            chunks.append(wavs[i][torch.min(chunk_anchors)-chunk_len : torch.min(chunk_anchors)]) 
        # chunks.append(wavs[i].squeeze(0)[torch.max(chunk_anchors):torch.max(chunk_anchors) + chunk_len ])
        else:
            wav_padded = torch.nn.functional.pad(wavs[i],(0,chunk_len-tmp_len),'constant',0)
            padded_lens[i] = tmp_len / chunk_len
            chunks.append(wav_padded) 

    chunks_all = torch.stack(chunks,dim=0)
    # actual_lens = torch.stack(actual_lens,dim=0)
    return chunks_all, padded_lens

def create_chunk_pairs(embedding_dict,trials):
    
    chunk_pairs = []
    labels = []
    for trial in trials:
        enrol_id = trial.split()[1]
        auth_id = trial.split()[2]
        target_id = trial.split()[0]


        enrol_emb = embedding_dict[enrol_id]
        auth_emb = embedding_dict[auth_id]

        chunk_pairs.append(torch.cat((enrol_emb,auth_emb),dim=0))
        labels.append(target_id)

    return torch.stack(chunk_pairs), torch.stack(labels)



# Compute embeddings from the waveforms
def compute_embedding(wavs, wav_lens):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """
    with torch.no_grad():
        feats = params["wav2vec2"](wavs)
        embeddings_all = params["encoder"](feats)
        if params["wav2vec2"].model.config.output_hidden_states:
            embeddings_frame = params["weighted_sum"](embeddings_all)
            embeddings = params["pooling"](embeddings_frame,wav_lens)
        else:
            embeddings = params["pooling"](embeddings_all,wav_lens)
    return embeddings.squeeze(1)


def compute_embedding_loop(data_loader,use_chunks=True):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """
    embedding_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, dynamic_ncols=True):
            batch = batch.to(params["device"])
            seg_ids = batch.id
            wavs, lens = batch.sig

            found = False
            for seg_id in seg_ids:
                if seg_id not in embedding_dict:
                    found = True
            if not found:
                continue
            wavs, lens = wavs.to(params["device"]), lens.to(params["device"])
            if use_chunks:
                chunks, lens = sample_chunks(wavs,lens)
                emb = compute_embedding(chunks, lens).unsqueeze(1)
            else:
                emb = compute_embedding(wavs,lens).unsqueeze(1)
            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()
    
        # chunk_pairs, labels = create_chunk_pairs(embedding_dict,trials)
    
    
    return embedding_dict

def get_classification_result(veri_test):
    """Computes positive and negative scores given the verification split."""

    positive_scores, negative_scores = [], []

    pairs = []
    labels = []
    with torch.no_grad():
        for i, line in enumerate(veri_test):

            # Reading verification file (enrol_file test_file label)
            lab_pair = int(line.split(" ")[0].rstrip().split(".")[0].strip())
            enrol_id = line.split(" ")[1].rstrip().split(".")[0].strip()
            test_id = line.split(" ")[2].rstrip().split(".")[0].strip()
            enrol = enrol_dict[enrol_id]
            test = test_dict[test_id]
            
            labels.append(lab_pair)
            pairs.append(torch.cat((enrol,test),dim=1))

            if lab_pair == 1:
                # positive_scores.append(torch.nn.functional.mse_loss(enrol,test,reduction='mean'))
                positive_scores.append(torch.nn.functional.cosine_similarity(enrol,test))
            else:
                # negative_scores.append(torch.nn.functional.mse_loss(enrol,test,reduction='mean'))
                negative_scores.append(torch.nn.functional.cosine_similarity(enrol,test))

        classification_pairs = torch.stack(pairs)
        
        classification_labels = torch.tensor(labels)
        classification_labels = classification_labels.to(params["device"])

        predictions = params["classifier"](classification_pairs.squeeze(1))
        
        # for i in range(predictions.size()[0]):
        #     if classification_labels[i] == 1.0:
        #         positive_scores.append(predictions[i][1])
        #     else:
        #         negative_scores.append(predictions[i][1])

        # targets = torch.nn.functional.one_hot(classification_labels,num_classes=2)
        error = classification_error(predictions,classification_labels)

    return error, positive_scores, negative_scores



def dataio_prep(params):
    "Creates the dataloaders and their data processing pipelines."

    data_folder = params["data_folder"]


    # Enrol data
    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["enrol_data"], replacements={"data_root": data_folder},
    )
    enrol_data = enrol_data.filtered_sorted(sort_key="duration",reverse=True)

    # Test data
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_data"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration",reverse=True)

    datasets = [enrol_data, test_data]

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # Set output
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    # Create dataloaders
    enrol_dataloader = sb.dataio.dataloader.make_dataloader(
        enrol_data, **params["enrol_dataloader_opts"]
    )
    test_dataloader = sb.dataio.dataloader.make_dataloader(
        test_data, **params["test_dataloader_opts"]
    )

    return  enrol_dataloader, test_dataloader


if __name__ == "__main__":
    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(
        params["save_folder"], os.path.basename(params["verification_file"])
    )
    download_file(params["verification_file"], veri_file_path)

    from voxceleb_prepare import prepare_voxceleb  # noqa E402

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Prepare data from dev of Voxceleb1
    prepare_voxceleb(
        data_folder=params["data_folder"],
        save_folder=params["save_folder"],
        verification_pairs_file=veri_file_path,
        splits=[ "test"],
        split_ratio=[100],
        seg_dur=10.0,
        source=params["voxceleb_source"]
        if "voxceleb_source" in params
        else None,
    )

    # here we create the datasets objects as well as tokenization and encoding
    enrol_dataloader, test_dataloader = dataio_prep(params)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected(params["device"])
    
    params["encoder"].eval()
    params["classifier"].eval()
    params["weighted_sum"].eval()
    params["pooling"].eval()

    params["wav2vec2"].to(params["device"])
    params["encoder"].to(params["device"])
    params["classifier"].to(params["device"])
    params["weighted_sum"].to(params["device"])
    params["pooling"].to(params["device"])
    # Computing  enrollment and test embeddings
    logger.info("Computing enroll/test embeddings...")

    # First run
    enrol_dict = compute_embedding_loop(enrol_dataloader,use_chunks=False)
    test_dict = compute_embedding_loop(test_dataloader,use_chunks=False)

    # if "score_norm" in params:
    #     train_dict = compute_embedding_loop(train_dataloader)  

    # Compute the EER
    logger.info("Computing Classification Error..")
    # Reading standard verification split
    with open(veri_file_path) as f:
        veri_test = [line.rstrip() for line in f]

    results, positive_scores, negative_scores = get_classification_result(veri_test)
    del enrol_dict, test_dict

    eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    logger.info("EER(%%)=%f", eer * 100)

    # min_dcf, th = minDCF(
    #     torch.tensor(positive_scores), torch.tensor(negative_scores)
    # )
    # logger.info("minDCF=%f", min_dcf * 100)
