# pip install bilibili-api-python
from bilibili_api import Credential, video
from datetime import datetime, timedelta, date
import asyncio
import argparse
from tqdm import tqdm
import os
from typing import List
import pandas as pd
import random
from pathlib import Path
from colorama import Fore, Back

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bvid', help="place where you hold all bvid files")
    parser.add_argument('--sess', help="place where you hold all sess files")
    parser.add_argument('--output', help="place where you hold processed csv")
    args = parser.parse_args()
    return args


def get_date_range(pubdate) -> List[date]:
    start_date = datetime.fromtimestamp(pubdate).date()
    end_date = datetime.now().date()
    delta = end_date - start_date
    date_list = [start_date + timedelta(days=i) for i in range(delta.days + 1)]
    return date_list

def read_bvids(bvid_location: Path) -> pd.DataFrame:
    return pd.read_csv(bvid_location)
    
def read_sess(SESSDATA_location: Path) -> pd.DataFrame: 
    return pd.read_csv(SESSDATA_location)
            
async def main():
    # Initialize Credential
    args = parser()
    bvid_location = args.bvid
    sess_location = args.sess
    output_folder = args.output

    bvids = read_bvids(bvid_location)
    sess_data = read_sess(SESSDATA_location=sess_location)
    sess_using_index = list(range(len(sess_data)))
    random.shuffle(sess_using_index)

    os.makedirs(output_folder, exist_ok=True)
    print('output folder created')
    
    for i in range(len(bvids)):
        bvid = bvids.loc[i,'BVID']
        if len(sess_using_index) == 0:
            sess_using_index = list(range(len(sess_data)))
            random.shuffle(sess_using_index)
        sess_data_indx = sess_using_index.pop()
        user_sess_cache = sess_data.loc[sess_data_indx,'SESSDATA']
        credential = Credential(sessdata=user_sess_cache)

        # Check if Credential is valid
        # is_valid = await credential.check_valid()
        # if not is_valid:
        #     print(Fore.RED, "Credential is invalid. Please check your SESSDATA.")
        #     print(Fore.RED, "This is your SESSDATA", user_sess_cache)
        #     print(Fore.RED, "This is your SESSDATA User", sess_data.loc[sess_data_indx,'USER NAME'])
        #     continue        
        

        try:
            # Get Video Info
            v = video.Video(bvid=bvid, credential=credential)
            info = await v.get_info()
            pubdate = info['pubdate']
            title = info['title']

            # Prepare storage directory
            movie_name = bvids.loc[i, 'Name']  # Get the movie name for the current bvid
            movie_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in movie_name)  # Make the name safe for file systems
            save_location = os.path.join(output_folder, f"{movie_name}.csv")  # Use the movie name for the CSV file

            print(Fore.GREEN, f"\nProgress: [{i+1}/{len(bvids)}], Downloading '{title}', data stored at: {save_location}")
            print(Fore.RESET)

            # Get Date Range
            date_list = get_date_range(pubdate)

            # Initialize list to store danmaku data
            danmaku_data = []

            # Fetch danmaku for each date
            for i, date_item in enumerate(tqdm(date_list, desc='Fetching danmaku')):
                if i % 10 != 0:
                    continue
                date_str = date_item.strftime('%Y-%m-%d')
                total_wait_time = 0
                wait_time = 1  # Initial wait time
                max_wait_time = 100  # Maximum total wait time
                success = False
                while total_wait_time <= max_wait_time and not success:
                    try:
                        # Fetch danmaku for the specific date
                        danmakus = await v.get_danmakus(page_index=0, date=date_item)
                        # Collect danmaku data
                        for dm in danmakus:
                            danmaku_data.append({
                                'movie_time': dm.dm_time,
                                'danmu': dm.text,
                                'sent_time': datetime.fromtimestamp(dm.send_time),
                                'user_id': dm.uid,
                                'user_id (CRC32)': dm.crc32_id,
                                'danmu_id': f"{dm.uid}_{dm.send_time}_{dm.dm_time}"  # Create a unique ID
                            })
                        success = True
                    except Exception as e:
                        print(f"Failed to fetch danmaku for {date_str}: {e}")
                        print(f"Waiting for {wait_time} seconds before retrying...")
                        await asyncio.sleep(wait_time)
                        total_wait_time += wait_time
                        wait_time *= 2  # Exponential backoff
                if not success:
                    print(f"Skipping date {date_str} after {total_wait_time} seconds of retries.")
                else:
                    # Wait for 1 second before next date
                    await asyncio.sleep(1)

            # Remove duplicates based on 'danmu_id'
            df = pd.DataFrame(danmaku_data)
            df.drop_duplicates(subset='danmu_id', inplace=True)
            
            # Additional duplicate removal based on content and timing
            df.drop_duplicates(subset=['danmu', 'movie_time', 'user_id'], inplace=True)

            # Save to CSV
            df.to_csv(save_location, index=False)
            print(f"Saved danmaku data to {save_location}")
        except Exception as e:
            print(f"An error occurred while processing BVID '{bvid}': {e}")

if __name__ == "__main__":
    asyncio.run(main())
