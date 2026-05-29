 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os

from comfy_api.latest import  io
import folder_paths
from .node_utils import  clear_comfyui_cache,tensor2image,audio2path
from .LongCat_Video.run_demo_avatar_single_audio_to_video import load_longcat_video_model,generate,get_audio_vocal,get_audio_emb,load_audio_vocal
from .LongCat_Video.run_demo_avatar_multi_audio_to_video import generate_multi
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

MAX_SEED = np.iinfo(np.int32).max
node_longcat_path = os.path.dirname(os.path.abspath(__file__))
weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir
weigths_longcat_current_path = os.path.join(folder_paths.models_dir, "longcat")
if not os.path.exists(weigths_longcat_current_path):
    os.makedirs(weigths_longcat_current_path)
folder_paths.add_model_folder_path("longcat", weigths_longcat_current_path) #  longcat dir


class LongCat_Video_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCat_Video_SM_Model",
            display_name="LongCat_Video_SM_Model",
            category="LongCat_Video",
            inputs=[
                io.Combo.Input("diffusion_models",options= ["none"] + folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf")),
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae")),
                io.Combo.Input("lora",options= ["none"] + folder_paths.get_filename_list("loras")),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, diffusion_models,gguf,vae,lora) -> io.NodeOutput:
        clear_comfyui_cache()
        dit_path=folder_paths.get_full_path("diffusion_models",diffusion_models) if diffusion_models != "none" else None
        gguf_path=folder_paths.get_full_path("gguf",gguf) if gguf != "none" else None
        vae_path=folder_paths.get_full_path("vae",vae) if vae != "none" else None
        lora_path=folder_paths.get_full_path("loras",lora) if lora != "none" else None
        model_path=dit_path or gguf_path
        model=load_longcat_video_model(model_path,vae_path, lora_path,node_longcat_path,use_int8=True if "int8" in model_path.lower() else None)
        return io.NodeOutput(model)
    

class LongCat_Video_SM_Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCat_Video_SM_Sampler",
            display_name="LongCat_Video_SM_Sampler",
            category="LongCat_Video",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("te_cond"),
                io.Conditioning.Input("au_cond"),
                io.Image.Input("image"),
                io.Combo.Input("stage_1",options= ['ai2v', 'at2v']),
                io.Combo.Input("resolution",options= ['480p', '720p']),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Int.Input("steps", default=8, min=1, max=500, step=1),
                io.Float.Input("text_guidance_scale", default=1.0, min=0.1, max=10.0, step=0.1, ),
                io.Float.Input("audio_guidance_scale", default=1.0, min=0.1, max=10.0, step=0.5,),
                io.Int.Input("ref_img_index", default=10, min=0, max=1024, step=1),
                io.Int.Input("mask_frame_range", default=3, min=0, max=1024, step=1),
                io.Int.Input("block_num", default=1, min=0, max=64,step=1),   
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )
    
    @classmethod
    def execute(cls, model,te_cond,au_cond,image,stage_1,resolution, seed, steps,text_guidance_scale,audio_guidance_scale,ref_img_index,mask_frame_range,block_num,) -> io.NodeOutput:
        clear_comfyui_cache()
        model.vae_to(device)
        if block_num==0:
            model.to(device)
        model.streaming_prefetch_count=block_num  if block_num > 0 else None
        if au_cond.get("left_full_audio_emb",None) is not None:
            image=generate_multi(model,au_cond,te_cond,device,seed,tensor2image(image),resolution,
                text_guidance_scale,audio_guidance_scale,steps,ref_img_index,mask_frame_range,model.use_distill)
        else:
            image=generate(model,au_cond,te_cond,device,seed,stage_1,tensor2image(image),resolution,
                text_guidance_scale,audio_guidance_scale,steps,ref_img_index,mask_frame_range,
                model.use_distill)
        if block_num==0:
            model.to("cpu")
            torch.cuda.empty_cache()
        return io.NodeOutput(image)

class LongCat_Video_SM_Encode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCat_Video_SM_Encode",
            display_name="LongCat_Video_SM_Encode",
            category="LongCat_Video",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt",default="A western man stands on stage under dramatic lighting, holding a microphone close to their mouth. Wearing a vibrant red jacket with gold embroidery, the singer is speaking while smoke swirls around them, creating a dynamic and atmospheric scene.",multiline=True),
                io.String.Input("negative_prompt",default="Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards.",multiline=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="te_cond"),
                ],
        )
    @classmethod
    def execute(cls, clip, prompt,negative_prompt,) -> io.NodeOutput: 
        tokens = clip.tokenize(prompt)
        prompt_embeds=clip.encode_from_tokens_scheduled(tokens)
        prompt_embeds=prompt_embeds[0][0]
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(1, 1, seq_len, -1)
        tokens = clip.tokenize(negative_prompt)
        negative_prompt_embeds=clip.encode_from_tokens_scheduled(tokens)
        negative_prompt_embeds=negative_prompt_embeds[0][0]
        _, seq_len, _ = negative_prompt_embeds.shape
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(1, 1, seq_len, -1)
        te_cond={"prompt_embeds":prompt_embeds.to(device,torch.bfloat16),"negative_prompt_embeds":negative_prompt_embeds.to(device,torch.bfloat16),"text":[prompt,negative_prompt],} # #torch.Size([1, 1, 512, 4096])
        clear_comfyui_cache()
       
        return io.NodeOutput(te_cond)
    
class LongCat_Video_SM_Audio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCat_Video_SM_Audio",
            display_name="LongCat_Video_SM_Audio",
            category="LongCat_Video",
            inputs=[
                io.AudioEncoder.Input("audio_encoder"),
                io.Audio.Input("audio"),
                io.Int.Input("save_fps", default=25, min=8, max=1024, step=1),
                io.Int.Input("num_segments", default=1, min=1, max=1024, step=1),
                io.Combo.Input("audio_type",options= ['para', 'add']),
                io.String.Input("p_box",default="",multiline=False),
                io.Audio.Input("left_audio",optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="au_cond"),
                ],
        )
    @classmethod
    def execute(cls, audio_encoder,audio,save_fps,num_segments,audio_type,p_box,left_audio=None) -> io.NodeOutput: 
        if p_box:
            import ast
            # 将类似 "[100, 80, 800, 640], [1001, 80, 800, 640]" 的字符串转为嵌套列表
            parsed_p_box = ast.literal_eval(f"[{p_box}]")
            assert isinstance(parsed_p_box, list) and len(parsed_p_box) >= 2 , "p_box must be a list of int ,and must lens >2"
        else:
            parsed_p_box = None

        au_cond=get_audio_emb(audio_encoder,audio,left_audio,audio_type,save_fps,num_segments,device,p_box=parsed_p_box)
        clear_comfyui_cache()
        return io.NodeOutput(au_cond)
    
class LongCat_Video_SM_Vocal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCat_Video_SM_Vocal",
            display_name="LongCat_Video_SM_Vocal",
            category="LongCat_Video",
            inputs=[
                io.AudioEncoder.Input("audio_encoder"),
                io.Audio.Input("audio"),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="audio_path"),
                ],
        )
    @classmethod
    def execute(cls, audio_encoder,audio,) -> io.NodeOutput: 
        audio_path,audio=get_audio_vocal(audio_encoder,audio2path(audio),folder_paths.get_output_directory())
        return io.NodeOutput(audio,audio_path)

class LongCat_Video_SM_WhisperModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCat_Video_SM_WhisperModel",
            display_name="LongCat_Video_SM_WhisperModel",
            category="LongCat_Video",
            inputs=[
                io.Combo.Input(
                    "audio_encoder",options=folder_paths.get_filename_list("audio_encoders") ,
                ),
            ],
            outputs=[
                io.AudioEncoder.Output(),
                ],
        )
    @classmethod
    def execute(cls, audio_encoder) -> io.NodeOutput: 
        a_checkpoint_path = folder_paths.get_full_path_or_raise("audio_encoders", audio_encoder)
        from .LongCat_Video.longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
        audio_encoder = get_audio_encoder(a_checkpoint_path, 'avatar-v1.5',os.path.join(node_longcat_path, "LongCat_Video/whisper-large-v3"))
        audio_feature_extractor = get_audio_feature_extractor(os.path.join(node_longcat_path, "LongCat_Video/whisper-large-v3"), 'avatar-v1.5')
        audio_encoder={"audio_encoder":audio_encoder,"audio_feature_extractor":audio_feature_extractor}
        return io.NodeOutput(audio_encoder)

class LongCat_Video_SM_VocalModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCat_Video_SM_VocalModel",
            display_name="LongCat_Video_SM_VocalModel",
            category="LongCat_Video",
            inputs=[
                io.Combo.Input(
                    "audio_encoder_vocal",options=["none"]+[i for i in folder_paths.get_filename_list("longcat") if i.endswith(".onnx")],
                ),
            ],
            outputs=[
                io.AudioEncoder.Output(),
                ],
        )
    @classmethod
    def execute(cls, audio_encoder_vocal) -> io.NodeOutput: 
        vocal_separator_path=folder_paths.get_full_path_or_raise("longcat", audio_encoder_vocal) if audio_encoder_vocal!="none" else None
        audio_encoder=load_audio_vocal(vocal_separator_path,folder_paths.get_output_directory(),weigths_longcat_current_path)
        return io.NodeOutput(audio_encoder)
    
