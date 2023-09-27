# from deepface import DeepFace

# img_path = "/var/wd_smit/localdata/Elon_Musk_Colorado_2022_(cropped2).jpg"
# models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "Dlib", "ArcFace"]
# embedding = DeepFace.represent(img_path, model_name=models[0])

# print(embedding[0]["facial_area"])

from deepface import DeepFace
import os
import pandas as pd

img_dir = "/var/wd_smit/internal_repos/backups/images"
models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "Dlib", "ArcFace"]
embeddings = []
face_coords = []
names = []

df = pd.DataFrame({"embedding": [], "facial_area": []})

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    embedding = DeepFace.represent(img_path, model_name=models[0])
    embeddings.append(embedding[0]["embedding"])
    face_coords.append(embedding[0]["facial_area"])

    dftemp = pd.DataFrame(embedding)
    df = pd.concat([df, dftemp], ignore_index=True)

    names.append(img_name)

df["Name"] = names

print(face_coords)
print(df.info)

# with open("/var/wd_smit/internal_repos/backups/embeddings_demo.txt", "w+") as f:
#     # write elements of list
#     for items in embeddings:
#         f.write("%s\n" % items)

#     print("File written successfully")
