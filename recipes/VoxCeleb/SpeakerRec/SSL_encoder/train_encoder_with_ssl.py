#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
import chunk
import os
import sys
import random
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import numpy as np
from pytorch_memlab import MemReporter, profile, set_target_gpu


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SSL_speaker(sb.core.Brain):
    """Class for speaker embedding training"
    """

    def random_choice(self,index,indices):
        tmp_select = torch.randint(0,len(indices),(1,))
        while torch.abs(tmp_select - index) < 2:
            tmp_select = torch.randint(0,len(indices),(1,))
        
        return tmp_select

    def sample_chunks(self,wavs,stage,lens):
        
        chunks = []
        indices = []

        chunk_len = int(self.hparams.chunk_size * self.hparams.sample_rate / 1000)
        lens = lens.detach().cpu().numpy()

        for i in range(self.hparams.batch_size):
            
            tmp_len = int(np.floor(lens[i] * wavs.size()[1]))

            assert tmp_len >= 2*chunk_len
            
            chunk_anchors = torch.randint(chunk_len,tmp_len-chunk_len,(2,1))
            
            chunks.append(wavs[i][torch.min(chunk_anchors)-chunk_len : torch.min(chunk_anchors)]) 
            
            if stage == sb.Stage.TRAIN:
                
                aug_select = torch.randperm(self.n_augment-1)[:2]
                chunks.append(wavs[i+self.hparams.batch_size * aug_select[0]][torch.min(chunk_anchors)-chunk_len : torch.min(chunk_anchors)]) 
                chunks.append(wavs[i+self.hparams.batch_size * aug_select[1]][torch.max(chunk_anchors):torch.max(chunk_anchors) + chunk_len ])
            else:
                chunks.append(wavs[i][torch.min(chunk_anchors)-chunk_len : torch.min(chunk_anchors)]) 
                chunks.append(wavs[i].squeeze(0)[torch.max(chunk_anchors):torch.max(chunk_anchors) + chunk_len ])

        chunks_all = torch.stack(chunks,dim=0)

        return chunks_all

            
    def sample_trials(self,indices):
        
        pairs = []
        
        # positive_pairs = [torch.tensor([i,i+1,1.0],dtype=torch.int32) for i in indices if i%2 == 0]
        
        # negative_pairs = []

        for i in indices:
            
            if i%2 == 0:
                pairs.append(torch.tensor([i,i+1,1.0],dtype=torch.int64))
                tmp_select = self.random_choice(i,indices)
                pairs.append(torch.tensor([i,tmp_select,0.0],dtype=torch.int64))
            # negative_pairs.append((i,tmp_select+ self.hparams.batch_size * torch.randint(0,len(self.hparams.augment_pipeline),(1,))))
            # negative_pairs.append(torch.tensor([i,tmp_select,0.0],dtype=torch.int32))

        return torch.stack(pairs)

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        if stage == sb.Stage.TRAIN:

            # Applying the augmentation pipeline
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            for count, augment in enumerate(self.hparams.augment_pipeline):

                # Apply augment
                wavs_aug = augment(wavs, lens)

                # Managing speed change
                if wavs_aug.shape[1] > wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else:
                    zero_sig = torch.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                if self.hparams.concat_augment:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs

            


            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            lens = torch.cat([lens] * self.n_augment)

        # Feature extraction and normalization
        
        chunks = self.sample_chunks(wavs,stage,lens)
        feats = self.modules.wav2vec(chunks)
        
        pairs = self.sample_trials(torch.arange(2*self.hparams.batch_size))
        pairs = pairs.to(self.device)

        encodings = self.modules.encoder(feats)
        if self.hparams.wav2vec2.model.config.output_hidden_states:
            encodings_w = self.modules.weighted_sum(encodings)
            encodings_pool = self.modules.pooling(encodings_w)
        else:
            encodings_pool = self.modules.pooling(encodings)

        enroll_encodings = encodings_pool.index_select(0,pairs[:,0])
        auth_encodings = encodings_pool.index_select(0,pairs[:,1])
    

        encoding_pairs = torch.cat((enroll_encodings,auth_encodings),dim=1)
        label_pairs = pairs[:,2]


        predictions = self.modules.classifier(encoding_pairs)
        
        # enroll_encodings = 


        # enroll_encodings = 

        # feats = self.modules.wav2vec(wavs)
        # feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + speaker classifier
        # embeddings = self.modules.embedding_model(feats)
        # outputs = self.modules.classifier(embeddings)

        # reporter = MemReporter()
        # reporter.report()

        return predictions, label_pairs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        predictions, label_pairs = predictions
        uttid = batch.id
        # spkid, _ = batch.spk_id_encoded

        # # Concatenate labels (due to data augmentation)
        # if stage == sb.Stage.TRAIN:
        #     spkid = torch.cat([spkid] * self.n_augment, dim=0)

        # loss = self.hparams.compute_cost(predictions, spkid, lens)
        targets = torch.nn.functional.one_hot(label_pairs,num_classes=2)
        loss = self.hparams.compute_cost(predictions, targets)


        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, label_pairs)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],num_to_keep=self.hparams.num_to_keep,
            )


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    train_data = train_data.filtered_sorted(
        sort_key="duration",reverse=True
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )


    valid_data = valid_data.filtered_sorted(
        sort_key="duration", reverse=True
    )

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        if hparams["random_chunk"]:
            duration_sample = int(duration * hparams["sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample)
            stop = start + snt_len_sample
        else:
            start = int(start)
            stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[train_data], output_key="spk_id",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data, valid_data, label_encoder


if __name__ == "__main__":

    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(
        hparams["save_folder"], os.path.basename(hparams["verification_file"])
    )
    download_file(hparams["verification_file"], veri_file_path)

    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from voxceleb_prepare import prepare_voxceleb  # noqa

    run_on_main(
        prepare_voxceleb,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "verification_pairs_file": veri_file_path,
            "splits": ["train", "dev"],
            "split_ratio": [90, 10],
            "random_segment": hparams["random_segment"],
            "seg_dur": 15.0,
            "max_dur": 15.0,
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, label_encoder = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    speaker_brain = SSL_speaker(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
