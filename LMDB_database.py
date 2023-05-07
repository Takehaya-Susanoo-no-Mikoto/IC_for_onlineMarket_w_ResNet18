import os
import pickle
import torch
from torchvision import transforms, datasets
import lmdb
from PIL import Image
import torch.utils.data as data

# Путь к папке с изображениями
image_folder = 'E:\\images\\train_1'

# Путь к базе данных LMDB
lmdb_path = 'E:\\pythonProject4\\images'

os.makedirs(lmdb_path, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 32

env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs=1)


with env.begin(write=True) as txn:
    # Перебор файлов в папке с изображениями
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            # Путь к файлу изображения
            image_path = os.path.join(root, file)

            # Загрузка изображения и метки класса
            image = Image.open(image_path).convert('RGB')
            label = os.path.basename(root)

            image_bytes = image.tobytes()

            # Добавление данных в базу данных LMDB
            txn.put(os.path.basename(image_path).encode('utf-8'), label.encode('utf-8'))


class LMDBDataset(data.Dataset):
    def __init__(self, lmdb_path, transform=None):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(buffers=True)
        self.length = self.txn.stat()['entries']
        self.transform = transform

    def __getitem__(self, index):
        key = f"{index:08d}".encode('ascii')
        value = self.txn.get(key)
        item = pickle.loads(value)
        image = item['image']
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.length


test_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.RandomRotation(30),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

test_set = LMDBDataset('lmdb_path', transform=test_transform)
test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)