import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import glob


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def tensor2im(var):
    # var shape: (3, H, W)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                fname = fname.split('.')[0]
                images.append((fname, path))
    return images


class ImagesDataset(Dataset):

    def __init__(
        self, 
        image_root, 
        name=None,
        source_transform=None, 
        c_root=None,
        w_root=None,
        mask_root=None, 
        lm_root=None,
        mode='jpg'
    ):
        self.source_paths = sorted(glob.glob(f'{image_root}/*.{mode}'))
        self.name = name
        self.source_transform = source_transform
        if self.source_transform is None:
            self.source_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.c_root = c_root
        self.mask_root = mask_root
        self.w_root = w_root
        self.lm_root = lm_root
        print('[NOTE]: Landmark is on the size of 256*256')
        
    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        img_path = self.source_paths[index]
        img = Image.open(img_path).convert('RGB').resize((512, 512))
        img = self.source_transform(img)
        
        fname = os.path.basename(img_path).split('.')[0]
        c = np.load(os.path.join(self.c_root, fname + '.npy')).astype(np.float32)

        data = {
            'img': img,
            'c': c,
            'fname': fname,
            'name': self.name
        }

        if self.w_root is not None:
            w = torch.load(os.path.join(self.w_root, fname + '.pt'))
            data['w'] = w

        if self.mask_root is not None:
            mask = torch.load(os.path.join(self.mask_root, fname + '.pt'))
            data['mask'] = mask
            
        if self.lm_root is not None:
            lm = torch.from_numpy(np.load(os.path.join(self.lm_root, fname + '.npy'))).float()
            data['lm'] = lm

        return data


class PTIDataset(Dataset):

    def __init__(
        self, 
        source_root, 
        source_transform=None, 
        c_root=None,
        w_root=None,
        mask_root=None, 
        lm_root=None,
        target_name='target',
        mode='jpg',
        dataset_block=None,
        output_root=None,
        select_range=None,
        filter_index=None,
    ):
        self.source_root = source_root
        self.source_transform = source_transform
        if self.source_transform is None:
            self.source_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.c_root = c_root
        self.mask_root = mask_root
        self.w_root = w_root
        self.lm_root = lm_root
        self.mode = mode
        self.target_name = target_name

        self.source_paths = sorted(glob.glob(f'{source_root}/*/'))
        
        if select_range is not None:
            self.source_paths = self.source_paths[:select_range]
        # bad_src = ['11055', '27065']
        if output_root is not None:
            total_number = len(self.source_paths)
            exist_paths = sorted(glob.glob(f'{output_root}/*.jpg'))
            residue_paths = []
            for src_path in self.source_paths:
                index = src_path.split('/')[-2]
                if not os.path.join(output_root, f'{index}.jpg') in exist_paths:
                    residue_paths.append(src_path)
            self.source_paths = residue_paths
            print(f'Total number: {total_number}; Finish number {len(exist_paths)}; Residual number: {len(self.source_paths)}')
        
        if dataset_block is not None:
            total_length = len(self.source_paths)
            dataset_block = dataset_block.split('/')
            index = int(dataset_block[0])
            total = int(dataset_block[1])
            block = total_length // total + 1
            st = (index - 1) * block
            ed = (index) * block
            self.source_paths = self.source_paths[st:ed]
            print(f'Dataset block: {index}/{total}; Image number: {st}-{ed}; Total number: {ed-st}')

        if filter_index is not None:
            self.source_paths = []
            for ff in filter_index:
                self.source_paths.append(os.path.join(source_root, f'{ff}/'))

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        path = self.source_paths[index]
        name = os.path.dirname(path).split('/')[-1]

        img_path = os.path.join(path, f'{self.target_name}.{self.mode}')
        img = Image.open(img_path).convert('RGB').resize((512, 512))
        img = self.source_transform(img)
        
        fname = self.target_name
        c = np.load(os.path.join(self.c_root, name, fname + '.npy')).astype(np.float32)

        data = {
            'img': img,
            'c': c,
            'fname': fname,
            'name': name
        }

        if self.w_root is not None:
            w = torch.load(os.path.join(self.w_root, name, fname + '.pt'))
            data['w'] = w

        if self.mask_root is not None:
            mask = torch.load(os.path.join(self.mask_root, name, fname + '.pt'))
            data['mask'] = mask
            
        if self.lm_root is not None:
            lm = torch.from_numpy(np.load(os.path.join(self.lm_root, name, fname + '.npy'))).float()
            data['lm'] = lm

        return data

    def build_inner_dataset(self, name):
        image_root = os.path.join(self.source_root, name)
        c_root = os.path.join(self.c_root, name)
        if self.w_root is not None:
            w_root = os.path.join(self.w_root, name)
        else:
            w_root = None
        if self.mask_root is not None:
            mask_root = os.path.join(self.mask_root, name)
        else:
            mask_root = None
        if self.lm_root is not None:
            lm_root = os.path.join(self.lm_root, name)
        else:
            lm_root = None
        dataset = ImagesDataset(
            image_root=image_root,
            name=name,
            source_transform=self.source_transform, 
            c_root=c_root,
            w_root=w_root,
            mask_root=mask_root, 
            lm_root=lm_root,
            mode=self.mode
        )
        return dataset



class PTIDataset_M(Dataset):

    def __init__(
        self, 
        source_root, 
        source_transform=None, 
        c_root=None,
        w_root=None,
        mask_root=None, 
        lm_root=None,
        target_name='target',
        mode='jpg'
    ):
        self.source_root = source_root
        self.source_transform = source_transform
        if self.source_transform is None:
            self.source_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.c_root = c_root
        self.mask_root = mask_root
        self.w_root = w_root
        self.lm_root = lm_root
        self.mode = mode
        self.target_name = target_name

        self.source_paths = sorted(glob.glob(f'{source_root}/*/'))
        temp = []
        for i in self.source_paths:
            if not i.endswith('m'):
                temp.append(i)

        self.source_paths = temp

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        path = self.source_paths[index]
        name = os.path.dirname(path).split('/')[-1]

        fname = self.target_name
        mname = fname + '_m'

        img_path = os.path.join(path, f'{self.target_name}.{self.mode}')
        img = Image.open(img_path).convert('RGB').resize((512, 512))
        img = self.source_transform(img)
        c = np.load(os.path.join(self.c_root, name, fname + '.npy')).astype(np.float32)

        data = {
            'img': img,
            'c': c,
            'fname': fname,
            'name': name,
        }

        mimg_path = os.path.join(path, f'{mname}.{self.mode}')
        if os.path.exists(mimg_path):
            mimg = Image.open(mimg_path).convert('RGB').resize((512, 512))
            mimg = self.source_transform(mimg)
            
            mc = np.load(os.path.join(self.c_root, name, mname + '.npy')).astype(np.float32)
            data['mimg'] = mimg
            data['mc'] = mc

            if self.w_root is not None:
                mw = torch.load(os.path.join(self.w_root, name, mname + '.pt'))
                data['mw'] = mw

            mmask = torch.load(os.path.join(self.mask_root, name, mname + '.pt'))
            data['mmask'] = mmask

            mlm = torch.from_numpy(np.load(os.path.join(self.lm_root, name, mname + '.npy'))).float()
            data['mlm'] = mlm

        if self.w_root is not None:
            w = torch.load(os.path.join(self.w_root, name, fname + '.pt'))
            data['w'] = w
            

        if self.mask_root is not None:
            mask = torch.load(os.path.join(self.mask_root, name, fname + '.pt'))
            data['mask'] = mask
            
            
        if self.lm_root is not None:
            lm = torch.from_numpy(np.load(os.path.join(self.lm_root, name, fname + '.npy'))).float()
            data['lm'] = lm
            

        return data

    def build_inner_dataset(self, name):
        image_root = os.path.join(self.source_root, name)
        c_root = os.path.join(self.c_root, name)
        if self.w_root is not None:
            w_root = os.path.join(self.w_root, name)
        else:
            w_root = None
        if self.mask_root is not None:
            mask_root = os.path.join(self.mask_root, name)
        else:
            mask_root = None
        if self.lm_root is not None:
            lm_root = os.path.join(self.lm_root, name)
        else:
            lm_root = None
        dataset = ImagesDataset(
            image_root=image_root,
            name=name,
            source_transform=self.source_transform, 
            c_root=c_root,
            w_root=w_root,
            mask_root=mask_root, 
            lm_root=lm_root,
            mode=self.mode
        )
        return dataset



class PureImagesDataset(Dataset):

    def __init__(self, source_root, source_transform=None):
        self.source_paths = sorted(make_dataset(source_root))
        self.source_transform = source_transform

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB').resize((512, 512))
        from_im = np.asarray(from_im).transpose(2, 0, 1)
        dir = os.path.dirname(from_path)
        c_name = os.path.join(dir, fname + '.npy')
        return from_im, c_name

