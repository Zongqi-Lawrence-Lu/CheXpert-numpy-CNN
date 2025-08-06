import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

df = pd.read_csv("Data/train.csv")
df.fillna(np.nan, inplace = True)
label_cols = df.columns[5:] # the start of tags

image_list = []
tag_list = []
count = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    image_path = "Data/" + row["Path"][20:]

    # Selection Condition
    if row["Frontal/Lateral"] == "Frontal" and row["Sex"] == "Male":
        try:
            count += 1
            img = Image.open(image_path).convert("L")  # grayscale
            img = img.resize((368, 320)) # further shrinking
            img_array = np.array(img, dtype=np.uint8)
            image_list.append(img_array)

            #sex = float(row["Sex"] == "Female")  # Female: 1, Male: 0
            #age = float(row["Age"]) if not pd.isna(row["Age"]) else np.nan

            labels = row[label_cols].to_numpy(dtype=np.float32)

            #tag = np.concatenate(([age, sex], labels))
            tag_list.append(labels)

        except Exception as e:
            print(f"Failed on image: {image_path} - {e}")
            continue

print("Have {} images in that fit the condition".format(count))
images_np = np.stack(image_list)
tags_np = np.stack(tag_list)

'''
Age, Sex, No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion,
Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax,
Pleural Effusion, Pleural Other, Fracture, Support Devices
'''

np.save("images", images_np)
np.save("tags", tags_np)

print(f"\nSaved {len(images_np)} images")
print(f"Saved {len(tags_np)} tag vectors")
