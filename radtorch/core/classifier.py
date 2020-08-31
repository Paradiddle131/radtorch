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
from .data_processor import *
from .feature_extractor import *



class Classifier(object):

    def __init__(self,
                input_data_dict,
                type='log_reg',
                interaction_terms=False,
                cv=True,
                stratified=True,
                num_splits=5,
                parameters={},
                random_state=100,
                **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.input_data_dict=input_data_dict
        self.type=type
        self.interaction_terms=interaction_terms
        self.cv=cv
        self.stratified=stratified
        self.num_splits=num_splits
        self.parameters=parameters
        self.random_state=random_state

        # Load input feature table
        self.feature_names=self.input_data_dict['train']['features_names']
        self.train_features=self.input_data_dict['train']['features']
        self.train_labels=np.array(self.input_data_dict['train']['labels'])
        self.test_features=self.input_data_dict['test']['features']
        self.test_labels=np.array(self.input_data_dict['test']['labels'])

        # Interaction Terms
        if self.interaction_terms:
            log('Creating Interaction Terms for Train Dataset.')
            self.train_features=self.create_interaction_terms(self.train_features)
            log('Creating Interaction Terms for Test Dataset.')
            self.test_features=self.create_interaction_terms(self.test_features)
            log('Interaction Terms Created Successfully.')

        # Create Classifier object
        self.classifier=self.create_classifier(**self.parameters)
        self.classifier_type=self.classifier.__class__.__name__

    def create_classifier(self, **kw):

        """
        Creates Classifier Object
        """

        if self.type not in SUPPORTED_CLASSIFIER:
          log('Error! Classifier type not supported. Please check documentation for supported classifier types.')
          pass
        elif self.type=='lin_reg':
          classifier=LinearRegression(n_jobs=-1, **kw)
        elif self.type=='log_reg':
          classifier=LogisticRegression(max_iter=10000,n_jobs=-1, **kw)
        elif self.type=='lasso':
          classifier=Lasso(max_iter=10000,**kw)
        elif self.type=='elasticnet':
          classifier=ElasticNet(max_iter=10000, **kw)
        elif self.type=='ridge':
          classifier=RidgeClassifier(max_iter=10000, **kw)
        elif self.type=='sgd':
          classifier=SGDClassifier(**kw)
        elif self.type=='knn':
          classifier=KNeighborsClassifier(n_jobs=-1,**kw)
        elif self.type=='decision_trees':
          classifier=tree.DecisionTreeClassifier(**kw)
        elif self.type=='random_forests':
          classifier=RandomForestClassifier(**kw)
        elif self.type=='gradient_boost':
          classifier=GradientBoostingClassifier(**kw)
        elif self.type=='adaboost':
          classifier=AdaBoostClassifier(**kw)
        elif self.type=='xgboost':
          classifier=XGBClassifier(**kw)
        return classifier

    def info(self):

        """
        Returns table of different classifier parameters/properties.
        """

        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        return info

    def run(self):

        """
        Runs Image Classifier Training.
        """

        self.scores=[]
        self.train_metrics=[]

        if self.cv:
          if self.stratified:
            kf=StratifiedKFold(n_splits=self.num_splits, shuffle=True, random_state=self.random_state)
            log('Training '+str(self.classifier_type)+ ' with '+str(self.num_splits)+' split stratified cross validation.')
          else:
            kf=KFold(n_splits=self.num_splits, shuffle=True, random_state=self.random_state)
            log('Training '+str(self.classifier_type)+ ' classifier with '+str(self.num_splits)+' splits cross validation.')
          split_id=0
          for train, test in tqdm(kf.split(self.train_features, self.train_labels), total=self.num_splits):
            self.classifier.fit(self.train_features.iloc[train], self.train_labels[train])
            split_score=self.classifier.score(self.train_features.iloc[test], self.train_labels[test])
            self.scores.append(split_score)
            log('Split '+str(split_id)+' Accuracy = ' +str(split_score))
            self.train_metrics.append([[0],[0],[split_score],[0]])
            split_id+=1
        else:
          log('Training '+str(self.type)+' classifier without cross validation.')
          self.classifier.fit(self.train_features, self.train_labels)
          score=self.classifier.score(self.test_features, self.test_labels)
          self.scores.append(score)
          self.train_metrics.append([[0],[0],[score],[0]])
        self.scores = np.asarray(self.scores )
        self.classes=self.classifier.classes_.tolist()
        log(str(self.classifier_type)+ ' model training finished successfully.')
        log(str(self.classifier_type)+ ' overall training accuracy: %0.2f (+/- %0.2f)' % ( self.scores .mean(),  self.scores .std() * 2))
        self.train_metrics = pd.DataFrame(data=self.train_metrics, columns = ['Train_Loss', 'Valid_Loss', 'Train_Accuracy', 'Valid_Accuracy'])
        self.trained_model = self.classifier
        return self.trained_model, self.train_metrics



    #
    # def average_cv_accuracy(self):
    #
    #     """
    #     Returns average cross validation accuracy.
    #     """
    #
    #     if self.cv:
    #       return self.scores.mean()
    #     else:
    #       log('Error! Training was done without cross validation. Please use test_accuracy() instead.', gui=gui)
    #
    # def test_accuracy(self) :
    #
    #     """
    #     Returns accuracy of trained classifier on test dataset.
    #     """
    #
    #     acc= self.classifier.score(self.test_features, self.test_labels)
    #     return acc
    #
    #
    #
    # def confusion_matrix(self,title='Confusion Matrix',cmap=None,normalize=False,figure_size=(8,6)):
    #
    #     """
    #     Displays confusion matrix using trained classifier and test dataset.
    #
    #     Parameters
    #     ----------
    #     - title (string, optional): name to be displayed over confusion matrix.
    #     - cmap (string, optional): colormap of the displayed confusion matrix. This follows matplot color palletes. default=None.
    #     - normalize (boolean, optional): normalize values. default=False.
    #     - figure_size (tuple, optional): size of the figure as width, height. default=(8,6)
    #
    #     """
    #
    #     pred_labels=self.classifier.predict(self.test_features)
    #     true_labels=self.test_labels
    #     cm = metrics.confusion_matrix(true_labels, pred_labels)
    #     show_confusion_matrix(cm=cm,
    #                           target_names=self.classes,
    #                           title=title,
    #                           cmap=cmap,
    #                           normalize=normalize,
    #                           figure_size=figure_size
    #                           )
    #
    # def roc(self, **kw):
    #
    #     """
    #     Display ROC and AUC of trained classifier and test dataset.
    #
    #     """
    #
    #     show_roc([self], **kw)

    def predict(self, input_image_path, all_predictions=False, **kw):

        """

        Description
        -----------
        Returns label prediction of a target image using a trained classifier. This works as part of pipeline only for now.


        Parameters
        ----------

        - input_image_path (string, required): path of target image.

        - all_predictions (boolean, optional): return a table of all predictions for all possible labels.


        """

        classifier=self.classifier

        transformations=self.data_processor.transformations

        model=self.feature_extractor.model

        if input_image_path.endswith('dcm'):
            target_img=dicom_to_pil(input_image_path)
        else:
            target_img=Image.open(input_image_path).convert('RGB')

        target_img_tensor=transformations(target_img)
        target_img_tensor=target_img_tensor.unsqueeze(0)

        with torch.no_grad():
            model.to('cpu')
            target_img_tensor.to('cpu')
            model.eval()
            out=model(target_img_tensor)
            out=out.tolist()
        image_features=pd.DataFrame(out, columns=self.feature_names)

        class_to_idx = self.data_processor.classes()

        if all_predictions:
            try:
                A = self.data_processor.classes().keys()
                B = self.data_processor.classes().values()
                C = self.classifier.predict_proba(image_features)[0]
                C = [("%.4f" % x) for x in C]
                return pd.DataFrame(list(zip(A, B, C)), columns=['LABEL', 'LABEL_IDX', 'PREDICTION_ACCURACY'])
            except:
                log('All predictions could not be generated. Please set all_predictions to False.')
                pass
        else:
            prediction=self.classifier.predict(image_features)

            return (prediction[0], [k for k,v in class_to_idx.items() if v==prediction][0])

    def export(self, output_path):

        """
        Exports the Classifier object for future use.

        Parameters
        ----------
        output_path (string, required): output file path.

        """
        try:
          outfile=open(output_path,'wb')
          pickle.dump(self,outfile)
          outfile.close()
          log('Classifier exported successfully.')
        except:
          raise TypeError('Error! Classifier could not be exported.')

    def export_trained_classifier(self, output_path):
        """
        Exports the trained classifier for future use.

        Parameters
        ----------
        output_path (string, required): output file path.

        """
        try:
          outfile=open(output_path,'wb')
          pickle.dump(self.classifier,outfile)
          outfile.close()
          log('Trained Classifier exported successfully.')
        except:
          raise TypeError('Error! Trained Classifier could not be exported.')

    # NEEDS TESTING
    def misclassified(self, num_of_images=4, figure_size=(5,5), table=False, **kw): # NEEDS CHECK FILE PATH !!!!!
      pred_labels=(self.classifier.predict(self.test_features)).tolist()
      true_labels=self.test_labels.tolist()
      accuracy_list=[0.0]*len(true_labels)

      y = copy.deepcopy(self.test_features)
      paths=[]
      for i in y.index.tolist():paths.append(self.test_feature_extractor.feature_table.iloc[i]['IMAGE_PATH'])  # <<<<< this line was changed .. check. / Accuracy not showing correctly !!

      misclassified_dict=misclassified(true_labels_list=true_labels, predicted_labels_list=pred_labels, accuracy_list=accuracy_list, img_path_list=paths)
      show_misclassified(misclassified_dictionary=misclassified_dict, transforms=self.data_processor.transformations, class_to_idx_dict=self.data_processor.classes(), is_dicom = self.is_dicom, num_of_images = num_of_images, figure_size =figure_size)
      misclassified_table = pd.DataFrame(misclassified_dict.values())
      if table:
          return misclassified_table

    # NEEDS TESTING
    def coef(self, figure_size=(50,10), plot=False):#BETA
      coeffs = pd.DataFrame(dict(zip(self.feature_names, self.classifier.coef_.tolist())), index=[0])
      if plot:
          coeffs.T.plot.bar(legend=None, figsize=figure_size);
      else:
          return coeffs

    # NEEDS TESTING
    def create_interaction_terms(self, table):#BETA
        self.interaction_features=table.copy(deep=True)
        int_feature_names = self.interaction_features.columns
        m=len(int_feature_names)
        for i in tqdm(range(m)):
            feature_i_name = int_feature_names[i]
            feature_i_data = self.interaction_features[feature_i_name]
            for j in range(i+1, m):
                feature_j_name = int_feature_names[j]
                feature_j_data = self.interaction_features[feature_j_name]
                feature_i_j_name = feature_i_name+'_x_'+feature_j_name
                self.interaction_features[feature_i_j_name] = feature_i_data*feature_j_data
        return self.interaction_features
