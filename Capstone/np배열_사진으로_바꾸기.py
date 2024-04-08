import numpy as np
import pandas as pd
from PIL import Image

csv_file = 'dataset.csv'

def save_images_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    for index, row in data.iterrows():
        state = row['state']
        image_data = eval(row['image'])
        image_array = np.array(image_data)
        image_array = image_array.reshape(26, 34)
        image = Image.fromarray(image_array.astype('uint8'), 'L')  # 'L'은 8비트 흑백 이미지를 나타냄

        if state == 'open':
            image.save(f'0/{index}.png')
        elif state == 'close':
            image.save(f'1/{index}.png')

save_images_from_csv(csv_file)
