import os
import json
import time
import math
import random
import uuid
import shutil
import PIL.Image
import numpy as np
from pathlib import Path

import torch
#import torch.distributed as dist

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.utils import load_image

from .longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline,get_audio_embedding_whisper,get_audio_embedding_whisper_
from .longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from .longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from .longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from .longcat_video.modules.quantization import load_quantized_dit,get_config
# from .longcat_video.context_parallel import context_parallel_util

# -------- avatar related --------
import librosa
from .longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
# from .longcat_video.audio_process.torch_utils import save_video_ffmpeg
from audio_separator.separator import Separator
from safetensors.torch import load_file as safe_load
from .utils import load_gguf_checkpoint, match_state_dict, set_gguf2meta_model
from accelerate import init_empty_weights

def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def generate_random_uid():
    timestamp_part = str(int(time.time()))[-6:]
    random_part = str(random.randint(100000, 999999))
    uid = timestamp_part + random_part
    return uid

def extract_vocal_from_speech(source_path, target_path, vocal_separator, audio_output_dir_temp):
    outputs = vocal_separator.separate(source_path)
    if len(outputs) <= 0:
        print("Audio separate failed. Using raw audio.")
        return None
        
    default_vocal_path = Path(os.path.join(audio_output_dir_temp ,"vocals", f"{outputs[0]}"))
    default_vocal_path = default_vocal_path.resolve().as_posix()
    # cmd = f"mv '{default_vocal_path}' '{target_path}'"
    # os.system(cmd)    
    shutil.move(default_vocal_path, target_path) 
    return target_path

def replace_to_vocal_suffix(raw_speech_path):
    """Replace the suffix of the raw speech path with _vocal.wav"""
    path = Path(raw_speech_path)
    new_name = path.stem + "_vocal" + path.suffix
    return str(path.with_name(new_name))

def load_longcat_video_model(model_path,vae_path,distill_checkpoint_path="", node_longcat_path="",
                             use_int8=False,model_type='avatar-v1.5',):

    use_distill=True if distill_checkpoint_path else False 

    # initialize models
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(node_longcat_path, 'LongCat_Video/LongCat-Video'), subfolder="tokenizer", torch_dtype=torch.bfloat16)
    
    vae_config=AutoencoderKLWan.load_config(os.path.join(node_longcat_path, 'LongCat_Video/LongCat-Video/vae/config.json'))
    vae=AutoencoderKLWan.from_config(vae_config, torch_dtype=torch.bfloat16)
    vae_sd=safe_load(vae_path, device="cpu")
    vae.load_state_dict(vae_sd, strict=False)
    vae=vae.eval().to(torch.bfloat16)
    del vae_sd
    #vae = AutoencoderKLWan.from_single_file(vae_path,config=os.path.join(node_longcat_path, 'LongCat_Video/LongCat-Video/vae/config.json'), torch_dtype=torch.bfloat16) 
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(node_longcat_path, 'LongCat_Video/LongCat-Video'), subfolder="scheduler", torch_dtype=torch.bfloat16)
    
    if model_path.endswith(".safetensors"):
        if use_int8:
            print("[INFO] Loading INT8 quantized DiT model...")
            dit = load_quantized_dit(os.path.join(node_longcat_path, 'LongCat_Video/LongCat-Video'), subfolder="base_model_int8",cp_split_hw=[1,1], single_file=model_path)
        else: 
            print("[INFO] Loading normal  DiT model...")
            config_path = os.path.join(node_longcat_path, 'LongCat_Video/LongCat-Video/dit')
            #config=LongCatVideoAvatarTransformer3DModel.load_config(config_path)
            config=get_config(config_path,cp_split_hw=[1,1])
            with torch.device('meta'):
                dit=LongCatVideoAvatarTransformer3DModel.from_config(**config)
            sd=safe_load(model_path, device="cpu")
            X=dit.load_state_dict(sd, strict=False,assign=True)
            del sd
            dit=dit.eval().to(torch.bfloat16)
            #dit.cp_split_hw=[1,1]
            print(X)  
            #dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(os.path.join(node_longcat_path, 'LongCat_Video/LongCat-Video'), subfolder="base_model", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)
    elif model_path.endswith(".gguf"): # TODO need to support gguf
        print("[INFO] Loading gguf model...")
        config_path = os.path.join(node_longcat_path, 'LongCat_Video/LongCat-Video/dit')
        #config_path = os.path.join(node_longcat_path, 'LongCat_Video/LongCat-Video/dit/config.json')
        #config=LongCatVideoAvatarTransformer3DModel.load_config(config_path)
        config=get_config(config_path,cp_split_hw=[1,1])
        with init_empty_weights():  
            dit=LongCatVideoAvatarTransformer3DModel(**config)
        sd=load_gguf_checkpoint(model_path)      
        match_state_dict(dit, sd,show_num=10)
        set_gguf2meta_model(dit,sd,torch.bfloat16,torch.device("cpu"),) 
        del sd
        #dit.cp_split_hw=[1,1]      
    else:
        raise ValueError(f"Unsupported model format: {model_path}")
    if use_distill:
        if os.path.exists(distill_checkpoint_path):
            dit.load_lora(distill_checkpoint_path, "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
            dit.enable_loras(["dmd"])
    # initialize pipeline
    pipe = LongCatVideoAvatarPipeline(
        tokenizer = tokenizer,
        text_encoder = None,
        vae = vae,
        scheduler = scheduler,
        dit = dit,
        audio_encoder=None,
        audio_feature_extractor=None,
        model_type=model_type
    )
    #pipe.to(local_rank)
    pipe.use_distill = use_distill
    return pipe

def audio_prepare_multi(left_speech_array, right_speech_array, generate_duration,  sr=16000, audio_type='para'):
    #left_speech_array, right_speech_array = None, None
    # if left_temp_vocal_path is not None:
    #     left_speech_array, sr = librosa.load(left_temp_vocal_path, sr=sample_rate)
    #     left_raw_speech_array, _ = librosa.load(left_raw_speech_path, sr=sample_rate) 
    
    # if right_temp_vocal_path is not None:
    #     right_speech_array, sr = librosa.load(right_temp_vocal_path, sr=sample_rate)
    #     right_raw_speech_array, _ = librosa.load(right_raw_speech_path, sr=sample_rate) 
    
    if left_speech_array is None:
        left_speech_array = np.zeros_like(right_speech_array)
        #left_raw_speech_array = np.zeros_like(right_raw_speech_array)
    
    if right_speech_array is None:
        right_speech_array = np.zeros_like(left_speech_array)
        #right_raw_speech_array = np.zeros_like(left_raw_speech_array)
    
    if audio_type == 'add':
        left_speech_array_ext = np.concatenate([left_speech_array, np.zeros_like(right_speech_array)])
        right_speech_array_ext = np.concatenate([np.zeros_like(left_speech_array), right_speech_array])
        #merge_raw_speech = np.concatenate([left_raw_speech_array, np.zeros_like(right_raw_speech_array)]) + np.concatenate([np.zeros_like(left_raw_speech_array), right_raw_speech_array])
    elif audio_type == 'para':
        left_speech_array_ext = left_speech_array
        right_speech_array_ext = right_speech_array
        #merge_raw_speech = left_raw_speech_array + right_raw_speech_array
    else:
        raise NotImplementedError(f"Unsupported audio_type of {audio_type}")
    
    assert len(left_speech_array_ext) == len(right_speech_array_ext), f"The two speech lengths should be equal"
   

    source_duraion = len(left_speech_array_ext) / sr
    added_sample_nums = math.ceil((generate_duration - source_duraion) * sr)
    if added_sample_nums > 0:
        left_speech_array_ext  = np.append(left_speech_array_ext, [0.]*added_sample_nums)
        right_speech_array_ext = np.append(right_speech_array_ext, [0.]*added_sample_nums)

    return left_speech_array_ext, right_speech_array_ext

def prepare_audio(audio, sample_rate=16000):
    speech_array = np.array(audio["waveform"].squeeze(0), dtype=np.float32)
    sr = audio["sample_rate"]
    if sr != 16000:
        import librosa
        speech_array = librosa.resample(speech_array, orig_sr=sr, target_sr=sample_rate)
        sr = 16000
    return speech_array,sr

def get_audio_emb(audio_encoder,audio,left_audio,audio_type,save_fps,num_segments,device,p_box,model_type='avatar-v1.5' ):
    num_frames=93
    num_cond_frames = 13
    audio_stride=1
     # audio padding to target length
    generate_duration = num_frames / save_fps + (num_segments-1)*(num_frames-num_cond_frames) / save_fps
    
    speech_array, sr = prepare_audio(audio)
    if left_audio is not None:
        left_speech_array, _ = prepare_audio(left_audio)
    else:
        left_speech_array = None
    left_person_bbox, right_person_bbox,back_full_audio_emb,left_full_audio_emb,other_person_bbox = None, None, None, None,None
    use_background_silent_audio = False
    if p_box is not None:
        # bbox: [left_y_min, left_x_min, left_y_max, left_x_max]
        # x and y coordinates correspond to the width and height dimensions, respectively
        left_person_bbox=p_box[0]
        right_person_bbox=p_box[1]
        other_person_bbox = p_box[2] if len(p_box) > 2 else None
        use_background_silent_audio = other_person_bbox is not None and len(other_person_bbox) > 0

    if left_speech_array is not None:
        left_speech_array_ext, right_speech_array_ext = audio_prepare_multi(left_speech_array,speech_array, generate_duration, sr=sr, audio_type=audio_type)
        left_full_audio_emb = get_audio_embedding_whisper_(audio_encoder, left_speech_array_ext, fps=save_fps*audio_stride, )
        full_audio_emb = get_audio_embedding_whisper_(audio_encoder, right_speech_array_ext, fps=save_fps*audio_stride,)
        if torch.isnan(left_full_audio_emb).any() or torch.isnan(full_audio_emb).any():
            raise ValueError(f"broken audio embedding with nan values")
        if use_background_silent_audio:
            back_full_audio_emb = get_audio_embedding_whisper_(audio_encoder,np.zeros_like(left_speech_array_ext), fps=save_fps*audio_stride, )
        assert left_full_audio_emb.shape == full_audio_emb.shape, f"Inconsistent audio embedding shape."
        if use_background_silent_audio:
            assert left_full_audio_emb.shape == back_full_audio_emb.shape, f"Inconsistent audio embedding shape between speaker and background."
    else:
        source_duraion = len(speech_array) / sr
        added_sample_nums = math.ceil((generate_duration - source_duraion) * sr)
        if added_sample_nums > 0:
            speech_array = np.append(speech_array, [0.]*added_sample_nums)
        full_audio_emb=get_audio_embedding_whisper_(audio_encoder, speech_array, fps=save_fps*audio_stride, ) #torch.Size([2142, 5, 1280])
    if torch.isnan(full_audio_emb).any():
        raise ValueError(f"broken audio embedding with nan values") 

    au_cond={
        "full_audio_emb": full_audio_emb,
        "num_segments": num_segments,
        "audio_stride": audio_stride,
        "back_full_audio_emb": back_full_audio_emb,
        "left_full_audio_emb": left_full_audio_emb,
        "left_person_bbox": left_person_bbox,
        "right_person_bbox": right_person_bbox,
        "other_person_bbox": other_person_bbox,
        "use_background_silent_audio": use_background_silent_audio
    }
    return au_cond


def load_audio_vocal(vocal_separator_path,audio_output_dir_temp,checkpoint_dir):
    if vocal_separator_path is None:
        vocal_separator_path = os.path.join(checkpoint_dir, 'Kim_Vocal_2.onnx')
    os.makedirs(audio_output_dir_temp, exist_ok=True)
    audio_output_dir_temp = Path(audio_output_dir_temp)
    audio_separator_model_path = os.path.dirname(vocal_separator_path)
    audio_separator_model_name = os.path.basename(vocal_separator_path)
    vocal_separator = Separator(
        output_dir=audio_output_dir_temp / "vocals",
        output_single_stem="vocals",
        model_file_dir=audio_separator_model_path,
    )

    vocal_separator.load_model(audio_separator_model_name)
    vocal_separator.onnx_execution_provider = ["CUDAExecutionProvider"]
    return vocal_separator


def get_audio_vocal(vocal_separator,raw_speech_path,audio_output_dir_temp,):
    
    vocal_path=replace_to_vocal_suffix(raw_speech_path)
    os.makedirs(os.path.dirname(vocal_path), exist_ok=True)

    temp_vocal_path = extract_vocal_from_speech(raw_speech_path,vocal_path , vocal_separator, audio_output_dir_temp)       
    import librosa
    vocal_array, sr = librosa.load(temp_vocal_path, sr=16000)
    #print("vocal_array.shape", vocal_array.shape)
    audio={
        "waveform": torch.from_numpy(vocal_array).unsqueeze(0).unsqueeze(0),
        "sample_rate": sr,
    }
    return temp_vocal_path,audio

def generate(pipe,condition,te_cond,device,seed,stage_1,cond_image,resolution,
             text_guidance_scale,audio_guidance_scale,num_inference_steps,ref_img_index,mask_frame_range,
             use_distill,model_type='avatar-v1.5'):
    num_frames=93 # 滑动窗口13,硬编码为93确保滑动窗口覆盖整个视频
    num_cond_frames=13
    audio_stride=condition['audio_stride']
    full_audio_emb=condition['full_audio_emb']
    num_segments=condition['num_segments']

    # prepare audio embedding for the first clip
    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames
    center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0]-1)
    audio_emb = full_audio_emb[center_indices][None,...].to(device)
    
    if use_distill and model_type == "avatar-v1.5":
        num_inference_steps = 8
        text_guidance_scale = 1.0
        audio_guidance_scale = 1.0

    if resolution == '480p':
        height, width = 480, 832
    elif resolution == '720p':
        height, width = 768, 1280
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)


    #if local_rank == 0:
    print(f"Generating {stage_1} 1/{num_segments}...")

    if stage_1 == 'at2v':
        # ==============================
        #          at2v (480P)
        # ==============================
        output_tuple = pipe.generate_at2v(
            prompt=None,
            negative_prompt=None,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type='both',
            audio_emb=audio_emb,
            use_distill=use_distill,
            prompt_embeds=te_cond["prompt_embeds"], 
            negative_prompt_embeds=te_cond["negative_prompt_embeds"], 
            text=te_cond["text"],
        )
        output, latent = output_tuple 
        output = output[0] 
        video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        video = [PIL.Image.fromarray(img) for img in video]

        #if cp_rank == 0:
        output_tensor = torch.from_numpy(np.array(video)).float() / 255.0
        #save_video_ffmpeg(output_tensor, os.path.join(output_dir, "at2v_demo_1"), raw_speech_path, fps=save_fps, quality=5)
        del output
        torch_gc()
    
    elif stage_1 == 'ai2v':
        # ==============================
        #          ai2v (480P)
        # ==============================
        image_path =cond_image # input_data['cond_image']
        image = load_image(image_path)
        output_tuple = pipe.generate_ai2v(
            image=image,
            prompt=None,
            negative_prompt=None,
            resolution=resolution,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            output_type='both',
            generator=generator,
            audio_emb=audio_emb,
            use_distill=use_distill,
            prompt_embeds=te_cond["prompt_embeds"], 
            negative_prompt_embeds=te_cond["negative_prompt_embeds"], 
            text=te_cond["text"],
        )
        output, latent = output_tuple
        output = output[0]
        video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        video = [PIL.Image.fromarray(img) for img in video]

        #if cp_rank == 0:
        output_tensor = torch.from_numpy(np.array(video)).float() / 255.0
        #save_video_ffmpeg(output_tensor, os.path.join(output_dir, "ai2v_demo_1"), raw_speech_path, fps=save_fps, quality=5)
        del output
        torch_gc()
    else:
        raise NotImplementedError(f"Not supported type of stage_1: {stage_1}")

    # if context_parallel_util.get_cp_size() > 1:
    #     torch.distributed.barrier(group=context_parallel_util.get_cp_group())

    # =========================================
    #         long video generation (480P)
    # =========================================
    # load parsed long video args
    #ref_img_index = args.ref_img_index
    #mask_frame_range = args.mask_frame_range

    width, height = video[0].size
    current_video = video
    ref_latent = latent[:, :, :1].clone()
    all_generated_frames = video

    for segment_idx in range(1, num_segments):
        #if local_rank == 0:
        print(f"Generating segment {segment_idx+1}/{num_segments}...")
        
        # prepare audio embedding for the next clip
        audio_start_idx = audio_start_idx + audio_stride * (num_frames - num_cond_frames)
        audio_end_idx   = audio_start_idx + audio_stride * num_frames
        center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0]-1)
        audio_emb = full_audio_emb[center_indices][None,...].to(device)
        
        output_tuple = pipe.generate_avc(
            video=current_video,
            video_latent=latent, 
            prompt=None,
            negative_prompt=None,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type='both',
            use_kv_cache=True,
            offload_kv_cache=True,
            enhance_hf=True if not use_distill else False,
            audio_emb=audio_emb,
            ref_latent=ref_latent,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range,
            use_distill=use_distill,
            prompt_embeds=te_cond["prompt_embeds"], 
            negative_prompt_embeds=te_cond["negative_prompt_embeds"], 
            text=te_cond["text"],
        )
        output, latent = output_tuple

        output = output[0]
        new_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        del output

        all_generated_frames.extend(new_video[num_cond_frames:])

        current_video = new_video

        #if cp_rank == 0:
        output_tensor = torch.from_numpy(np.array(all_generated_frames)/255.0).float()
        # save_video_ffmpeg(output_tensor, os.path.join(output_dir, f"video_continue_{segment_idx+1}"), raw_speech_path, fps=save_fps, quality=5)
        # del output_tensor
    return output_tensor

