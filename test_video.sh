python test_video.py --rate_num 4 --test_config test_video_config.json \
    --cuda 1 --cuda_idx 0 -w 1 \
    --save_decoded_frame 1 \
    --stream_path /path/to/reconstructions \
    --output_path /path/to/output/json/file \
    --model_path_i /path/to/GLC_image \
    --model_path_p /path/to/GLC_video
