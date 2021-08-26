#!/usr/bin/env python
# coding: utf-8

import argparse

from nemo.utils import logging
from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE
from nemo.collections.asr.parts.utils.diarization_utils import (
        dump_json_to_file,
        write_txt,
        get_uniq_id_from_audio_path,
        get_file_lists,
)



"""
Supported ASR models: QuartzNet15x5Base

CH109: All sessions have two speakers.

python get_json_ASR_and_diarization.py \
--audiofile_list_path='/disk2/scps/audio_scps/callhome_ch109.scp' \
--reference_rttmfile_list_path='/disk2/scps/rttm_scps/callhome_ch109.rttm' \
--oracle_num_speakers=2

AMI: Oracle number of speakers in EN2002c.Mix-Lapel is 3, not 4.

python get_json_ASR_and_diarization.py \
--audiofile_list_path='/disk2/datasets/amicorpus/mixheadset_test_wav.list' \
--reference_rttmfile_list_path='/disk2/datasets/amicorpus/mixheadset_test_rttm.list' \
--oracle_num_speakers=2

"""

CONFIG_URL = "https://raw.githubusercontent.com/NVIDIA/NeMo/stable/examples/speaker_recognition/conf/speaker_diarization.yaml"

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_speaker_model", type=str, help="Fullpath of the Speaker embedding extractor model (*.nemo).", required=True)
parser.add_argument("--audiofile_list_path", type=str, help="Fullpath of a file contains the list of audio files", required=True)
parser.add_argument("--reference_rttmfile_list_path", default=None, type=str, help="Fullpath of a file contains the list of rttm files")
parser.add_argument("--oracle_vad_manifest", default=None, type=str, help="External VAD file for diarization")
parser.add_argument("--oracle_num_speakers", help="Either int or text file that contains number of speakers")
parser.add_argument("--threshold", default=50, type=int, help="Threshold for ASR based VAD")
parser.add_argument("--diar_config_url", default=CONFIG_URL, type=str, help="Config yaml file for running speaker diarization")
parser.add_argument("--csv", default='result.csv', type=str, help="")
args = parser.parse_args()


params = {
    "time_stride": 0.02, # This should not be changed if you are using QuartzNet15x5Base.
    "offset": -0.18, # This should not be changed if you are using QuartzNet15x5Base.
    "round_float": 2,
    "window_length_in_sec": 1.5,
    "shift_length_in_sec": 0.75,
    "round_float": 2,
    "print_transcript": False,
    "lenient_overlap_WDER": True, #False,
    "threshold": args.threshold,  # minimun width to consider non-speech activity
    "external_oracle_vad": True if args.oracle_vad_manifest else False,
    "diar_config_url": args.diar_config_url,
    "ASR_model_name": 'QuartzNet15x5Base-En',
}

asr_diar_offline = ASR_DIAR_OFFLINE(params)

asr_model = asr_diar_offline.set_asr_model(params['ASR_model_name']) 

asr_diar_offline.create_directories()

audio_file_list = get_file_lists(args.audiofile_list_path)

transcript_logits_list = asr_diar_offline.run_ASR(asr_model, audio_file_list)

word_list, spaces_list, word_ts_list = asr_diar_offline.get_speech_labels_list(transcript_logits_list, audio_file_list)

oracle_manifest = asr_diar_offline.write_VAD_rttm(asr_diar_offline.oracle_vad_dir, audio_file_list) if not args.oracle_vad_manifest else args.oracle_vad_manifest

asr_diar_offline.run_diarization(audio_file_list, oracle_manifest, args.oracle_num_speakers, args.pretrained_speaker_model)

if args.reference_rttmfile_list_path:

    ref_rttm_file_list = get_file_lists(args.reference_rttmfile_list_path)

    diar_labels, ref_labels_list, DER_result_dict = asr_diar_offline.eval_diarization(audio_file_list, ref_rttm_file_list)

    total_riva_dict = asr_diar_offline.write_json_and_transcript(
                                            audio_file_list,
                                            transcript_logits_list,
                                            diar_labels,
                                            word_list,
                                            word_ts_list,
                                            spaces_list,
                                        )

    
    WDER_dict = asr_diar_offline.get_WDER(total_riva_dict, DER_result_dict, audio_file_list, ref_labels_list)
    
    effective_WDER = asr_diar_offline.get_effective_WDER(DER_result_dict, WDER_dict)

    logging.info(f" total \nWDER : {WDER_dict['total']:.4f} \
                          \nDER  : {DER_result_dict['total']['DER']:.4f} \
                          \nFA   : {DER_result_dict['total']['FA']:.4f} \
                          \nMISS : {DER_result_dict['total']['MISS']:.4f} \
                          \nCER  : {DER_result_dict['total']['CER']:.4f} \
                          \nspk_counting_acc : {DER_result_dict['total']['spk_counting_acc']:.4f} \
                          \neffective_WDER : {effective_WDER:.4f}")

    asr_diar_offline.write_result_in_csv(args, WDER_dict, DER_result_dict, effective_WDER)


