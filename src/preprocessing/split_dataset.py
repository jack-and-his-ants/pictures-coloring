from sklearn.model_selection import train_test_split
import os

train_split_sample = 0.8
val_split_sample = 0.1
test_split_sample = 0.1

src_path = "dataset/L_luminance"
all_files = os.listdir(src_path)
dest_path = "splits/"

train_split,test_split = train_test_split(all_files,test_size=test_split_sample+val_split_sample,random_state=42)
test_split,val_split = train_test_split(test_split,test_size=val_split_sample/(val_split_sample+test_split_sample),random_state=42)

with open(dest_path+"train_split.txt",'w') as f:
    for file in train_split:
        f.write(file+"\n")

with open(dest_path+"test_split.txt",'w') as f:
    for file in test_split:
        f.write(file+"\n")

with open(dest_path+"validation_split.txt",'w') as f:
    for file in val_split:
        f.write(file+"\n")

