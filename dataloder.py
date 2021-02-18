from glob import glob
import numpy as np 
import cv2 
import functools
import image_data_aug


def one_hot(labels, num_classes):
    num_labels = len(labels)
    i = np.arange(num_labels) * num_classes
    one_hot = np.zeros((num_labels, num_classes))
    one_hot.flat[i + np.array(labels)] = 1
    return one_hot


class DataLoader:

    def __init__(self, data_dir, categories_rule):
        self.epoch = 0
        self.data_dir = data_dir
        self.categories_rule = categories_rule

        self.data_dict = self._build_img_dict_multi_dirs()
        self.data = self._update_egs()


    def _update_egs(self):
        data = list(self.data_dict.keys())
        np.random.shuffle(data)
        return data


    def _build_img_dict(self):
        '''
        input: [image directory,        category1
                image directory, ...]   category2
        '''      
        _dict = {}
        categories = []
        for idx, rule in enumerate(self.categories_rule):
            categories.append(0)
            images = glob(self.data_dir + rule)
            for image in images:
                _dict[image] = idx
                categories[idx] += 1

        print('{}'.format(categories))
        return _dict


    def _build_img_dict_multi_dirs(self):
        '''
        input: [[image directories, ...],  category1
                [image directories, ...]]  category2
        '''
        _dict = {}
        categories = []
        for idx, rules in enumerate(self.categories_rule):
            categories.append(0)
            images = []
            for r in rules:
                images += glob(self.data_dir + r)
            
            for image in images:
                _dict[image] = idx
                categories[idx] += 1
        
        print('{}'.format(categories))
        return _dict


    def load_one_image(self, image_dir):
        im = cv2.imread(image_dir)
        #im = im[:,:,::-1]
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        im_rgb = cv2.resize(im_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        im_rgb = im_rgb.astype('float32')

        im_rgb /= 127.5
        im_rgb -= 1.
        return im_rgb


    def next_batch(self, batch_size):
        batch_x, batch_y = [], []
        for i in range(batch_size):
            img_dir = self.data.pop()
            batch_x.append(self.load_one_image(img_dir))
            batch_y.append(self.data_dict[img_dir])

            if not self.data:
                self.epoch += 1
                self.data = self._update_egs()

        return np.array(batch_x), one_hot(batch_y, len(self.categories_rule))



class DataLoaderAug(DataLoader):
    def __init__(self, data_dir, categories_aug_rule):
        super().__init__(data_dir, categories_aug_rule)


    def _augmentation(self, image, aug_type):
        
        tokens = aug_type.split('#')
        for augmentation in tokens:
            if augmentation == 'flip0':
                func = functools.partial(
                    image_data_aug.flip,
                    axis = 0
                )
                image = func(image)
            elif augmentation == 'flip1':
                func = functools.partial(
                    image_data_aug.flip,
                    axis = 1
                )
                image = func(image)
            elif augmentation == 'random_rotation':
                func = functools.partial(
                    image_data_aug.random_rotation,
                ) 
                image = func(image)
            elif augmentation == 'random_blur':
                func = functools.partial(
                    image_data_aug.random_blur,
                    p = 0.7
                ) 
                image = func(image)

        return image


    def _build_img_dict_multi_dirs(self):
        '''
        input:  [
                    [
                        [ dir1, ratio, types, contain_original_image],   -> image directory and augmentation rules
                        [ dir2, ... ],
                        ...
                    ]      -> category 1 
                    [...], -> category n-1
                    [...]  -> category n
                ]
        '''
        _dict = {}
        categories = []

        for idx_category, category in enumerate(self.categories_rule):
            categories.append(0)
            images = []

            for image_dir_info in category:
                images_dir = glob(self.data_dir + image_dir_info[0])

                # add original image to list
                if image_dir_info[3]:
                    images += images_dir
                    
                # shuffle for random select image
                np.random.shuffle(images_dir)

                for img in images_dir[:int(len(images_dir) * image_dir_info[1] + 1)]:
                    img = '{aug_type}###{img_dir}'.format(
                        aug_type = image_dir_info[2],
                        img_dir = img
                    )
                    images.append(img)
            
            for image in images:
                _dict[image] = idx_category
                categories[idx_category] += 1

        print('{}'.format(categories))
        return _dict



    def load_one_image(self, image_dir):
        im = None
        tokens = image_dir.split('###')

        if len(tokens) == 1:
            im = cv2.imread(image_dir)
        
        else:
            im = cv2.imread(tokens[1])
            im = self._augmentation(im, tokens[0])
        
        #im = im[:,:,::-1]
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        im_rgb = cv2.resize(im_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        im_rgb = im_rgb.astype('float32')

        im_rgb /= 127.5
        im_rgb -= 1.
        return im_rgb

