"""Transparent arithmetic, not measured GPU/model memory or speed."""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024**2
GIB = 1024**3


def tensor_bytes(batch, channels, height, width, element_bytes):
    return batch*channels*height*width*element_bytes


def attention_bytes(batch, heads, queries, keys, element_bytes):
    return batch*heads*queries*keys*element_bytes


def round_float32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def round_float16(x):
    return struct.unpack('<e', struct.pack('<e', x))[0]


def round_bfloat16(x):
    if not math.isfinite(x):
        raise ValueError('Worksheet accepts finite coordinates')
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', rounded))[0]


def haversine_m(lat1, lon1, lat2, lon2):
    p1,p2=map(math.radians,(lat1,lat2))
    dp,dl=math.radians(lat2-lat1),math.radians(lon2-lon1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*6371008.8*math.asin(math.sqrt(min(1.,a)))


def report():
    current=(tensor_bytes(1,6,96,96,4)+tensor_bytes(1,2,384,384,4)
             +tensor_bytes(1,5,192,192,4)+tensor_bytes(1,8,96,96,4)
             +tensor_bytes(1,3,256,256,1))
    history=2*tensor_bytes(1,6,96,96,4)
    lat,lon=60.11,-120.24
    coordinates=[]
    for name,fn in [('float64',float),('float32',round_float32),('float16',round_float16),('bfloat16',round_bfloat16)]:
        la,lo=fn(lat),fn(lon)
        coordinates.append({'dtype':name,'latitude':la,'longitude':lo,
                            'ground_displacement_m':haversine_m(lat,lon,la,lo)})
    return {
        'kind':'analytical_component_arithmetic_not_gpu_profile',
        'memory_estimates_exclude':['saved backward activations','convolution workspaces','autocast caches',
                                   'allocator overhead','CUDA context','display/other applications','framework temporaries'],
        'input_payload':{'current_tile_mib':current/MIB,'two_extra_thermal_frames_mib':history/MIB,
                         'with_two_extra_frames_mib':(current+history)/MIB,
                         '100000_tiles_with_history_gib':100000*(current+history)/GIB},
        'fp32_adam_weights_gradients_two_moments_mib':{str(n):n*16/MIB for n in (2000000,5000000,10000000)},
        'one_384_square_64channel_feature_mib':{
            'batch1_bf16':tensor_bytes(1,64,384,384,2)/MIB,
            'batch2_bf16':tensor_bytes(2,64,384,384,2)/MIB,
            'batch1_fp32':tensor_bytes(1,64,384,384,4)/MIB},
        'materialized_attention_scores_only_gib':{
            '256sq_queries_96sq_keys_8heads_bf16_batch1':attention_bytes(1,8,256**2,96**2,2)/GIB,
            '256sq_queries_384sq_keys_8heads_bf16_batch1':attention_bytes(1,8,256**2,384**2,2)/GIB,
            '256sq_self_attention_1head_fp32_batch1':attention_bytes(1,1,256**2,256**2,4)/GIB},
        'dense_256sq_by_256sq_float32_distance_or_transport_matrix_gib':256**4*4/GIB,
        'coordinate_cast_example':{'original_latitude':lat,'original_longitude':lon,'rounding':coordinates,
                                   'interpretation':'quantization displacement only, not satellite geolocation accuracy'},
        'absolute_temperature_300K_spacing_K':{'float16':.25,'bfloat16':2.,
            'interpretation':'representable spacing at 300K; compute contrasts/normalization in FP32 before AMP'},
        'hypothetical_time_hours_100k_tiles_30epochs':{str(rate):100000*30/rate/3600 for rate in (5,20,50)},
        'throughput_values_are':'hypothetical processed tiles per second; no GPU speed claim',
        'no_native_model_trained':True}


if __name__=='__main__':
    value=report()
    target=ROOT/'results/memory_worksheet.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(value,indent=2))
