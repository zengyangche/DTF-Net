import os
import shutil


def categorize_files(source_folder, destination_folder):
   
    folders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]

    for folder in folders:
        folder_path = os.path.join(source_folder, folder)
        files = os.listdir(folder_path)

        for file in files:
         
            category = file.split('_')[-1]

           
            destination_category_folder = os.path.join(destination_folder, category)
            os.makedirs(destination_category_folder, exist_ok=True)

        
            source_file_path = os.path.join(folder_path, file)
            destination_file_path = os.path.join(destination_category_folder, file)
            shutil.move(source_file_path, destination_file_path)

            print(f"Moved {file} to {destination_category_folder}")


 
source_folder_path_train = "../dataset/brats20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"  
destination_folder_path_train = "../dataset/BraTS2020/BraTS2020_Training_Data"   

source_folder_path_val = "../dataset/brats20/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"   
destination_folder_path_val = "../dataset/BraTS2020/BraTS2020_Validation_Data"   
 
 
categorize_files(source_folder_path_train, destination_folder_path_train)
categorize_files(source_folder_path_val, destination_folder_path_val)


