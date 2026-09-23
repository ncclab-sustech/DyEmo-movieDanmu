import chardet
import os
import pandas as pd
import numpy as np
import random
import re

def convert_to_seconds(time_str):
    if time_str.count(':') == 1:
        time_str = '00:' + time_str  
    try:
        return pd.to_timedelta(time_str).total_seconds()  
    except Exception:
        return np.nan  
    
def read_group_danmu(file_dir, file, n_danmu_per_group, movie_time_name='movie_time', 
                     danmu_name='danmu', is_number=True,seed=42):
    rng = random.Random(seed)
    danmu_dict = {}
    with open(os.path.join(file_dir, file), 'rb') as f:
        encoding_type = chardet.detect(f.read())['encoding']

    print(file)
    try:
        df = pd.read_csv(os.path.join(file_dir, file), encoding='gbk')
        print('use gbk encoding')
    except:
        df = pd.read_csv(os.path.join(file_dir, file), encoding=encoding_type)
        print('use %s encoding' % encoding_type)

    print(file, 'danmu number:', len(df))
    result = []
    time_ranges = []

    # first_element = df[movie_time_name].iloc[0]
    # print(first_element)
    # is_number = bool(re.match(r'^\d+(\.\d+)?$', first_element))

    if is_number:
        df[movie_time_name] = pd.to_numeric(df[movie_time_name], errors='coerce')
        df = df[np.isfinite(df[movie_time_name])]
        df.reset_index(drop=True, inplace=True)
    else:
        df[movie_time_name] = df[movie_time_name].apply(convert_to_seconds).astype(int)

    df[movie_time_name] = df[movie_time_name].apply(lambda x: float(x))
    df = df.sort_values(by=movie_time_name)
    grouped = df.groupby(df[movie_time_name].astype(int))

    buffer_texts = []
    buffer_times = []

    for time, group in grouped:
        texts = group[danmu_name].tolist()
        times = group[movie_time_name].tolist()

        if (len(texts) > n_danmu_per_group) and (len(buffer_texts)==0):
            sampled_indices = rng.sample(range(len(texts)), n_danmu_per_group)
            sampled_texts = [str(texts[i]) for i in sampled_indices]   # add str()
            sampled_times = [times[i] for i in sampled_indices]

            result.append('\n'.join(sampled_texts))
            time_ranges.append((min(sampled_times), max(sampled_times)))
        else:
            buffer_texts.extend(texts)
            buffer_times.extend(times)

            while len(buffer_texts) >= n_danmu_per_group:
                sampled_indices = rng.sample(range(len(buffer_texts)), n_danmu_per_group)
                combined_texts = [str(buffer_texts[i]) for i in sampled_indices]
                combined_times = [buffer_times[i] for i in sampled_indices]
                
                result.append('\n'.join(combined_texts))
                time_ranges.append((min(combined_times), max(combined_times)))

                buffer_texts = []
                buffer_times = []

    if buffer_texts:
        result.append('\n'.join(buffer_texts))
        time_ranges.append((min(buffer_times), max(buffer_times)))

    danmu_dict[file] = result
    danmu_dict[file+'_time_range'] = time_ranges
    return danmu_dict
    
