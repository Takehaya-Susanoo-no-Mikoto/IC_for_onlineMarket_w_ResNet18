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

def read_image(path):
    """Reads an image from a file."""
    with Image.open(path) as img:
        img = img.convert('RGB')
        img = np.array(img)
    return img

def read_data(data_dir):
    """Reads images and labels from a directory."""
    data = []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for file_name in os.listdir(class_dir):
            if not file_name.endswith('.jpg'):
                continue
            file_path = os.path.join(class_dir, file_name)
            img = read_image(file_path)
            label = class_name.encode()
            data.append((img, label))
    return data

def write_data(data, path):
    """Writes data to an Lmdb database."""
    map_size = 1024 * 1024 * 1024
    env = lmdb.open(path, map_size=map_size)
    with env.begin(write=True) as txn:
        for i, (img, label) in enumerate(data):
            key = f'{i:08}'.encode()
            txn.put(key, img)
            txn.put(key + b'_label', label)

data = read_data(image_folder)
write_data(data, lmdb_path)


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
