# ComfyUI_LongCat_Avatar
[LongCat_Avatar](https://github.com/meituan-longcat/LongCat-Video),an upgraded open-source framework for audio-driven human video generation.

Update
----
* add node   
* 部署节点，目前单人和双人测试通过，部分参数未严谨测试，请自行测试，有bug可以提issues或反馈给我的B站或小红书smthem账号

1.Installation  
----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_LongCat_Avatar
```
2.requirements  
----
* need [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) you need find a wheel
* 注意需要Block-Sparse-Attention，你需要找个轮子，或者花点时间自己编译 使用[官方](https://github.com/mit-han-lab/Block-Sparse-Attention) 或者我的[ smthemex/Block-Sparse-Attention](https://github.com/smthemex/Block-Sparse-Attention)
* Block-Sparse-Attention 轮子 [Block-Sparse-Attention](https://huggingface.co/smthem/LongCat-Video-Avatar-1.5-merge)  
```
pip install -r requirements.txt
```
3.checkpoints 
----
  
links: [dit-int8](https://huggingface.co/smthem/LongCat-Video-Avatar-1.5-merge)  
links: [text_encoders](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/text_encoders)  
links: [vae/vocal_separator/whisper-large-v3/lora](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5/tree/main)

```
├── ComfyUI/models/diffusion_models/
|     ├── LongCat-Video-Avatar-1.5-int8.safetensors
├── ComfyUI/models/loras/
|     ├── LongCat_Avatar_1.5_lora.safetensors
├── ComfyUI/models/vae/
|     ├── LongCat_Avatar_1.5_vae.safetensors
├── ComfyUI/models/clip/
|     ├── umt5_xxl_fp8_e4m3fn_scaled.safetensors
├── ComfyUI/models/longcat/ # 懒得写节点
|     ├── vocal_separator
|         ├── 4 files
|     ├── whisper-large-v3
|         ├── 13 files ,model.safetensors,config.json,tokenizer.json,vocab.json.. #模型只下载model.safetensors即可,json要下
```

4 Example
----

![](https://github.com/smthemex/ComfyUI_LongCat_Avatar/blob/main/example_workflows/example.png)

5 Citation
----

```
@misc{meituanlongcatteam2025longcatvideotechnicalreport,
      title={LongCat-Video Technical Report}, 
      author={Meituan LongCat Team and Xunliang Cai and Qilong Huang and Zhuoliang Kang and Hongyu Li and Shijun Liang and Liya Ma and Siyu Ren and Xiaoming Wei and Rixu Xie and Tong Zhang},
      year={2025},
      eprint={2510.22200},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.22200}, 
}
@misc{meituanlongcatteam2025longcatvideoavatar15technicalreport,
      title={LongCat-Video-Avatar 1.5 Technical Report}, 
      author={Meituan LongCat Team},
      year={2026},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={}, 
}
@misc{meituanlongcatteam2025longcatvideoavatartechnicalreport,
      title={LongCat-Video-Avatar Technical Report}, 
      author={Meituan LongCat Team},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={}, 
}
```
