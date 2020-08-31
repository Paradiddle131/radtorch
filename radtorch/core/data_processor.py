# Copyright (C) 2020 RADTorch and Mohamed Elbanan, MD
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/

# Update: 8/30/2020

from ..settings import *
from ..utils import *



class DataLoader(Dataset):

    def __init__(
                self,
                table,
                transformations,
                data_directory=None,
                is_path=True,
                is_dicom=False,
                mode='RAW',
                wl=None,
                image_path_column='IMAGE_PATH',
                image_label_column='IMAGE_LABEL',
                sampling=1.0,
                **kwargs):

        self.table=table
        self.transformations=transformations
        self.data_directory=data_directory
        self.is_dicom=is_dicom
        self.mode=mode
        self.wl=wl
        self.image_path_column=image_path_column
        self.image_label_column=image_label_column
        self.is_path=is_path
        self.sampling=sampling


        for k, v in kwargs.items():
            setattr((self, k, v))


        if isinstance(self.table, str):
            self.table=pd.read_csv(self.table)


        if self.is_path==False:
            if self.data_directory==None:
                log ('No data_directory was provided. Please check.')
                pass
            else:
                files=[]
                for i, r in self.table.iterrows():
                    files.append(self.data_directory+r[self.image_path_column])
                self.table[self.image_path_column]=pd.Series(files, index=self.table.index)


        if self.is_dicom: self.dataset_files=[x for x in (self.table[self.image_path_column].tolist()) if x.endswith('.dcm')]
        else: self.dataset_files=[x for x in (self.table[self.image_path_column].tolist()) if x.endswith(IMG_EXTENSIONS)]


        self.classes= list(self.table[self.image_label_column].unique())
        self.class_to_idx=class_to_idx(self.classes)


        if len(self.dataset_files)==0: log ('Error! No data files found in directory:'+ self.data_directory)
        if len(self.classes)==0:log ('Error! No classes extracted from directory:'+ self.data_directory)


    def __getitem__(self, index):
        """
        Handles how to get an image of the dataset.
        """
        image_path=self.table.iloc[index][self.image_path_column]
        if self.is_dicom:
            image=dicom_to_narray(image_path, self.mode, self.wl)
            image=Image.fromarray(image)
        else:
            image=Image.open(image_path).convert('RGB')
        image=self.transformations(image)
        label=self.table.iloc[index][self.image_label_column]
        label_idx=[v for k, v in self.class_to_idx.items() if k == label][0]
        return image, label_idx, image_path



    def __len__(self):
        """
        Returns number of images in dataset.
        """
        return len(self.dataset_files)

    def info(self):
        """
        Returns information of the dataset.
        """
        return show_dataset_info(self)

    def classes(self):
        """
        returns list of classes in dataset.
        """
        return self.classes

    def class_to_idx(self):
        """
        returns mapping of classes to class id (dictionary).
        """
        return self.class_to_idx

    def parameters(self):
        """
        returns all the parameter names of the dataset.
        """
        return self.__dict__.keys()

    def balance(self, method='upsample'):
        """
        Retuns a balanced dataset. methods={'upsample', 'downsample'}
        """
        return balance_dataset(dataset=self, label_col=self.image_label_column, method=method)

    def mean_std(self):
        """
        calculates mean and standard deviation of dataset.
        """
        self.mean, self.std= calculate_mean_std(torch.utils.data.DataLoader(dataset=self))
        return tuple(self.mean.tolist()), tuple(self.std.tolist())

    def normalize(self, **kwargs):
        """
        Returns a normalized dataset with either mean/std of the dataset or a user specified mean/std in the form of ((mean, mean, mean), (std, std, std)).
        """
        if 'mean' in kwargs.keys() and 'std' in kwargs.keys():
            mean=kwargs['mean']
            std=kwargs['std']
        else:
            mean, std=self.mean_std()
        normalized_dataset=copy.deepcopy(self)
        normalized_dataset.transformations.transforms.append(transforms.Normalize(mean=mean, std=std))
        return normalized_dataset

class Data_Processor():

    def __init__(
                self,
                data_directory,
                is_dicom=False,
                table=None,
                image_path_column='IMAGE_PATH',
                image_label_column='IMAGE_LABEL',
                is_path=True,
                mode='RAW',
                wl=None,
                balance_class=False,
                balance_class_method='upsample',
                normalize=((0,0,0), (1,1,1)),
                batch_size=16,
                num_workers=0,
                sampling=1.0,
                custom_resize=False,
                model_arch='resnet50',
                type='nn_classifier',
                transformations='default',
                extra_transformations=None,
                test_percent=0.2,
                valid_percent=0.2,
                device='auto',
                label_source='parentfolder',
                num_labels=1,
                missing_class='Unclassified',
                **kwargs):

        self.data_directory=data_directory
        self.is_dicom=is_dicom
        self.table=table
        self.image_path_column=image_path_column
        self.image_label_column=image_label_column
        self.is_path=is_path
        self.mode=mode
        self.wl=wl
        self.balance_class=balance_class
        self.balance_class_method=balance_class_method
        self.normalize=normalize
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.sampling=sampling
        self.custom_resize=custom_resize
        self.model_arch=model_arch
        self.type=type
        self.transformations=transformations
        self.extra_transformations=extra_transformations
        self.device=device
        self.test_percent=test_percent
        self.valid_percent=valid_percent
        self.label_source=label_source
        self.num_labels=num_labels
        self.missing_class=missing_class


    # Set Device to be used
        if self.device=='auto':
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.collate_function=None


    # Create Initial Master Table
        if isinstance(self.table, str):
            if self.table!='':
                self.table=pd.read_csv(self.table)
        elif isinstance(self.table, pd.DataFrame):
            self.table=self.table
        else:self.table=create_data_table(directory=self.data_directory,
                                            is_dicom=self.is_dicom,
                                            image_label_column=self.image_label_column,
                                            image_path_column=self.image_path_column,
                                            label_source=self.label_source,
                                            num_labels=self.num_labels,
                                            missing_class=self.missing_class)


    # Sampling to be used
        if isinstance (self.sampling, float):
            if self.sampling > 1.0 :
                log('Error! Sampling cannot be more than 1.0.')
                pass
            elif self.sampling == 0:
                log ('Error! Sampling canot be Zero.')
                pass
            else:
                self.table=self.table.sample(frac=self.sampling, random_state=100)
        else:
            log ('Error! Sampling is not float')
            pass

    # Data Transformations
        # 1- Determine which image size to use for Resize
        if self.custom_resize in [False, '', 0, None]: self.resize=model_dict[self.model_arch]['input_size']
        elif isinstance(self.custom_resize, int): self.resize=self.custom_resize
        else: log ('Image Custom Resize not allowed. Please recheck values specified.')

        # 2- Resize and convert single channel DICOM to 3 channel (Grayscale)
        if self.transformations=='default':
            if self.is_dicom:
                self.transformations=transforms.Compose([
                        transforms.Resize((self.resize, self.resize)),
                        transforms.transforms.Grayscale(3),
                        transforms.ToTensor()])
            else:
                self.transformations=transforms.Compose([
                    transforms.Resize((self.resize, self.resize)),
                    transforms.ToTensor()])


        # 3- Normalize Training Dataset (Notice that normalization is done only on train dataset using mean and standard deviation. No data-leaks to test dataset.)
        self.train_transformations=copy.deepcopy(self.transformations)
        if self.extra_transformations != None :
            for i in self.extra_transformations:
                self.train_transformations.transforms.insert(1, i)
        if isinstance (self.normalize, tuple):
            mean, std=self.normalize
            if len(mean) != 3 or len(std) != 3:
                log ('Error! Shape of supplied mean and/or std does not equal 3 for a 3 channel input data. Please check that the mean/std follow the following format: ((mean, mean, mean), (std, std, std))')
                pass
            else:
                self.train_transformations.transforms.append(transforms.Normalize(mean=mean, std=std))
        elif self.normalize!=False:
            log('Error! Selected mean and standard deviation are not allowed.')
            pass




    # Creating Dataset/DataLoader Objects
        self.master_dataset=DataLoader( data_directory=self.data_directory,
                                        table=self.table,
                                        is_dicom=self.is_dicom,
                                        mode=self.mode,
                                        wl=self.wl,
                                        image_path_column=self.image_path_column,
                                        image_label_column=self.image_label_column,
                                        is_path=self.is_path,
                                        sampling=1.0,
                                        transformations=self.transformations,)
        self.master_dataloader=torch.utils.data.DataLoader(dataset=self.master_dataset,batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_function)
        self.num_output_classes=len(self.master_dataset.classes)


        if self.type=='nn_classifier':
            self.temp_table, self.test_table=train_test_split(self.table, test_size=self.test_percent, random_state=100, shuffle=True)
            self.train_table, self.valid_table=train_test_split(self.temp_table, test_size=(len(self.table)*self.valid_percent/len(self.temp_table)), random_state=100, shuffle=True)
            if self.balance_class:
                self.train_table=balance_dataframe(dataframe=self.train_table, method=self.balance_class_method, label_col=self.image_label_column)
            self.train_dataset=DataLoader(  data_directory=self.data_directory,
                                            table=self.train_table,
                                            is_dicom=self.is_dicom,
                                            mode=self.mode,
                                            wl=self.wl,
                                            image_path_column=self.image_path_column,
                                            image_label_column=self.image_label_column,
                                            is_path=self.is_path,
                                            sampling=1.0,
                                            transformations=self.train_transformations,)

            self.valid_dataset=DataLoader(  data_directory=self.data_directory,
                                            table=self.valid_table,
                                            is_dicom=self.is_dicom,
                                            mode=self.mode,
                                            wl=self.wl,
                                            image_path_column=self.image_path_column,
                                            image_label_column=self.image_label_column,
                                            is_path=self.is_path,
                                            sampling=1.0,
                                            transformations=self.transformations,)

            self.test_dataset=DataLoader(   data_directory=self.data_directory,
                                            table=self.test_table,
                                            is_dicom=self.is_dicom,
                                            mode=self.mode,
                                            wl=self.wl,
                                            image_path_column=self.image_path_column,
                                            image_label_column=self.image_label_column,
                                            is_path=self.is_path,
                                            sampling=1.0,
                                            transformations=self.transformations,)

            self.train_dataloader=torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_function)
            self.valid_dataloader=torch.utils.data.DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_function)
            self.test_dataloader=torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_function)

        else:
            self.train_table, self.test_table=train_test_split(self.table, test_size=self.test_percent, random_state=100, shuffle=True)
            if self.balance_class:
                self.train_table=balance_dataframe(dataframe=self.temp_table, method=self.balance_class_method, label_col=self.image_label_column)
            self.train_dataset=DataLoader(  data_directory=self.data_directory,
                                            table=self.train_table,
                                            is_dicom=self.is_dicom,
                                            mode=self.mode,
                                            wl=self.wl,
                                            image_path_column=self.image_path_column,
                                            image_label_column=self.image_label_column,
                                            is_path=self.is_path,
                                            sampling=1.0,
                                            transformations=self.train_transformations,)

            self.test_dataset=DataLoader(   data_directory=self.data_directory,
                                            table=self.test_table,
                                            is_dicom=self.is_dicom,
                                            mode=self.mode,
                                            wl=self.wl,
                                            image_path_column=self.image_path_column,
                                            image_label_column=self.image_label_column,
                                            is_path=self.is_path,
                                            sampling=1.0,
                                            transformations=self.transformations,)

            self.train_dataloader=torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_function)
            self.test_dataloader=torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_function)



    def classes(self):
        """
        Returns dictionary of classes/class_idx in data.
        """
        return self.master_dataset.class_to_idx

    def class_table(self):
        """
        Returns table of classes/class_idx in data.
        """
        return pd.DataFrame(list(zip(self.master_dataset.class_to_idx.keys(), self.master_dataset.class_to_idx.values())), columns=['Label', 'Label_idx'])

    def info(self):
        """
        Returns full information of the data processor object.
        """
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        info=info.append({'Property':'master_dataset_size', 'Value':len(self.master_dataset)}, ignore_index=True)
        for i in ['train_dataset', 'valid_dataset','test_dataset']:
            if i in self.__dict__.keys():
                info.append({'Property':i+'_size', 'Value':len(self.__dict__[i])}, ignore_index=True)
        return info

    def dataset_info(self, plot=True, figure_size=(500,300)):
        """
        Displays information of the data and class breakdown.

        Parameters
        -----------
        plot (boolean, optional): True to display data as graph. False to display in table format. default=True
        figure_size (tuple, optional): Tuple of width and lenght of figure plotted. default=(500,300)
        """

        info_dict={}
        info_dict['dataset']=show_dataset_info(self.master_dataset)
        info_dict['dataset'].style.set_caption('Overall Dataset')
        if 'type' in self.__dict__.keys():
            for i in ['train_dataset','test_dataset']:
                if i in self.__dict__.keys():
                    info_dict[i]= show_dataset_info(self.__dict__[i])
                    info_dict[i].style.set_caption(i)
            if self.type=='nn_classifier':
                if 'valid_dataset' in self.__dict__.keys():
                    info_dict['valid_dataset']= show_dataset_info(self.__dict__['valid_dataset'])
                    info_dict[i].style.set_caption('valid_dataset')

        if plot:
            plot_dataset_info(info_dict, plot_size= figure_size)
        else:
            for k, v in info_dict.items():
                print (k)
                display(v)

    def sample(self, figure_size=(10,10), show_labels=True, show_file_name=False, gui=False):
        """
        Displays a sample from the training dataset. Number of images displayed is the same as batch size.

        Parameters
        ----------
        figure_size (tuple, optional): Tuple of width and lenght of figure plotted. default=(10,10)
        show_label (boolean, optional): show labels above images. default=True
        show_file_names (boolean, optional): show file path above image. default=False


        """
        show_dataloader_sample(self.train_dataloader, figure_size=figure_size, show_labels=show_labels, show_file_name=show_file_name, gui=gui)

    def check_leak(self, show_file=False):
        """
        Checks possible overlap between train and test dataset files.

        Parameters
        ----------
        show_file (boolean, optional): display table of leaked/common files between train and test. default=False.

        """
        train_file_list=self.train_dataset.table[self.image_path_column]
        test_file_list=self.test_dataset.table[self.image_path_column]
        leak_files=[]
        for i in train_file_list:
            if i in test_file_list:
                leak_files.append(i)
        log('Data Leak Check: '+str(len(train_file_list))+' train files checked. '+str(len(leak_files))+' common files were found in train and test datasets.')
        if show_file:
            return pd.DataFrame(leak_files, columns='leaked_files')

    def export(self, output_path):
        """
        Exports the Dtaprocessor object for future use.

        Parameters
        ----------
        output_path (string, required): output file path.

        """
        try:
            outfile=open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            log('Data Processor exported successfully.')
        except:
            raise TypeError('Error! Data Processor could not be exported.')
