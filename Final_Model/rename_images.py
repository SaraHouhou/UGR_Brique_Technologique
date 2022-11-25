import os
os.getcwd()
collection = "C:/Users/shouhou/OneDrive - Capgemini/Documents/TinyHGR/vir"
for root, dirs, files in os.walk(collection):
    for j, folder in enumerate(dirs):
        for i,filenames in enumerate(os.listdir(os.path.join(collection, folder))):
           absname = os.path.join(os.path.join(collection, folder), filenames)
           newname = os.path.join(os.path.join(collection, folder), "vir"+folder+str(i)+ ".jpg")
           os.rename(absname, newname)

print('well done!')