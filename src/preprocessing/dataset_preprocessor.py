from skimage import io, color
from sklearn.cluster import MiniBatchKMeans
import joblib
import numpy as np
import random
import os
from PIL import Image
import config as config


class Dataset():
    def __init__(self,source,destination,width,height):
        self.source_path = source
        self.destination_path = destination
        self.width = width
        self.height = height
        

    def resize(self,image):
        resized_img = Image.fromarray(image).resize((self.width,self.height))
        resized_img = np.array(resized_img)
        return resized_img
        
    
    def rgb_to_cie_lab(self, rgb):
        rgb_norm = rgb/255.0
        return color.rgb2lab(rgb_norm)

    
    def run_kmeans(self,save_path,n_of_files, k=64,random_state=42,batch_size=1000,verbose=True):
        list_src_dir = os.listdir(self.source_path)
        if n_of_files > len(list_src_dir):
            n_of_files = len(list_src_dir)
            print(f"Number of files to large, set to {n_of_files}")
        kmeans_list = random.sample(list_src_dir,n_of_files)
        pixels = []
        for image_name in kmeans_list:
            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                image_rgb = io.imread(os.path.join(self.source_path,image_name))
                image_resized = self.resize(image_rgb)
                image_lab = self.rgb_to_cie_lab(image_resized)
                ab = image_lab[:,:,1:3]
                ab = ab.reshape(-1,2)
                pixels.append(ab)
        pixels = np.concatenate(pixels,axis=0)

        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=batch_size,
            verbose=verbose
        )
        kmeans.fit(pixels)
        joblib.dump(
            kmeans,
            save_path)
        return kmeans

    

    def create_dataset(self,kmeans_path):
        #TODO load kmeans 
        load_kmeans = joblib.load(kmeans_path)
        list_src_dir = os.listdir(self.source_path)
        y_folder_name = 'class_labels'
        x_folder_name = 'L_luminance'
        if not os.path.exists(os.path.join(self.destination_path,y_folder_name)):
            os.mkdir(os.path.join(self.destination_path,y_folder_name))
        if not os.path.exists(os.path.join(self.destination_path,x_folder_name)):
            os.mkdir(os.path.join(self.destination_path,x_folder_name))
        

        for image_name in list_src_dir:
            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                image_rgb = io.imread(os.path.join(self.source_path,image_name))
                image_resized = self.resize(image_rgb)
                # handling png alpha channel
                if image_rgb.shape[-1] == 4:
                    image_rgb = image_rgb[:, :, :3]

                image_lab = self.rgb_to_cie_lab(image_resized)
                l = image_lab[:,:,0]
                ab = image_lab[:,:,1:3]

                clustered_ab = load_kmeans.predict(ab.reshape(-1,2))
                clustered_ab = clustered_ab.reshape(l.shape)
                # saving some memory using uint16 instead of uint64
                clustered_ab = clustered_ab.astype(np.uint16)


                np.save(os.path.join(self.destination_path,x_folder_name,image_name.split('.')[0])+".npy",l)
                np.save(os.path.join(self.destination_path,y_folder_name,image_name.split('.')[0])+".npy",clustered_ab)
        print("done")
        


    
if __name__ == "__main__":
    source_dir = "/data/training/all"
    destination_dir = "/test_dataset"
    dataset = Dataset(source_dir, destination_dir, config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    dataset.run_kmeans('test_kmeans.pkl', config.KMEANS_CLASSES)
    dataset.create_dataset('test_kmeans.pkl')
    

    

    