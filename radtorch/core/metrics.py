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




def show_roc(classifier_list, figure_size=(700,400)):
    output_notebook()

    output = []
    p = figure(plot_width=figure_size[0], plot_height=figure_size[1], title=('Receiver Operating Characteristic'), tools=TOOLS, toolbar_location='below', tooltips=[('','@x'), ('','@y')])
    p.line([0, 0.5, 1.0], [0, 0.5, 1.0], line_width=1.5, line_color='#93D5ED', line_dash='dashed')

    ind = 0

    auc_list = []

    legend_items = []

    for i in classifier_list:
        if i.type in [x for x in SUPPORTED_CLASSIFIER if x != 'nn_classifier']:
            true_labels=i.test_labels
            predictions=i.classifier.predict(i.test_features)
        else: true_labels, predictions = calculate_nn_predictions(model=i.trained_model, target_data_set=i.test_dataset, device=i.device)
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions)
        auc = metrics.roc_auc_score(true_labels, predictions)
        x = p.line(fpr, tpr, line_width=2, line_color= COLORS2[ind])
        legend_items.append((('Model '+i.classifier_type+'. AUC = '+'{:0.4f}'.format((auc))),[x]))

        ind = ind+1
        auc_list.append(auc)

    legend = Legend(items=legend_items, location=(10, -20))
    p.add_layout(legend, 'right')

    p.legend.inactive_fill_alpha = 0.7
    p.legend.border_line_width = 0
    p.legend.click_policy="hide"
    p.xaxis.axis_line_color = '#D6DBDF'
    p.xaxis.axis_label = 'False Positive Rate (1-Specificity)'
    p.yaxis.axis_label = 'True Positive Rate (Senstivity)'
    p.yaxis.axis_line_color = '#D6DBDF'
    p.xgrid.grid_line_color=None
    p.yaxis.axis_line_width = 2
    p.xaxis.axis_line_width = 2
    p.xaxis.major_tick_line_color = '#D6DBDF'
    p.yaxis.major_tick_line_color = '#D6DBDF'
    p.xaxis.minor_tick_line_color = '#D6DBDF'
    p.yaxis.minor_tick_line_color = '#D6DBDF'
    p.yaxis.major_tick_line_width = 2
    p.xaxis.major_tick_line_width = 2
    p.yaxis.minor_tick_line_width = 0
    p.xaxis.minor_tick_line_width = 0
    p.xaxis.major_label_text_color = '#99A3A4'
    p.yaxis.major_label_text_color = '#99A3A4'
    p.outline_line_color = None
    p.toolbar.autohide = True

    show(p)

    return auc_list

def show_confusion_matrix(cm,target_names,title='Confusion Matrix',cmap=None,normalize=False,figure_size=(8,6)):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=figure_size)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()



class Metrics():
    def __init__(self, model, device='auto',
                **kwargs):
        self.model = model
        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.device=='auto': self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.model.trained_model:
            log('Error! Provided model was not trained/fitted. Please check.')
            pass

    def average_cv_accuracy(self):
        if self.model.cv:
          return self.model.scores.mean()
        else:
          log('Error! Training was done without cross validation. Please use test_accuracy() instead.', gui=gui)

    def test_accuracy(self) :
        return self.model.trained_model.score(self.model.test_features, self.model.test_labels)
        return acc

    def confusion_matrix(self,title='Confusion Matrix',cmap=None,normalize=False,figure_size=(8,6),target_dataset=None):
        if self.model.type != 'nn_classifier':
            pred_labels=self.model.trained_model.predict(self.model.test_features)
            true_labels=self.model.test_labels
            cm = metrics.confusion_matrix(true_labels, pred_labels)
            show_confusion_matrix(cm=cm,
                                  target_names=self.model.classes,
                                  title=title,
                                  cmap=cmap,
                                  normalize=normalize,
                                  figure_size=figure_size
                                  )
        else:
            if target_dataset==None:target_dataset=self.data_processor.test_dataset
            target_classes=(self.data_processor.classes()).keys()

            true_labels = []
            pred_labels = []

            self.model.trained_model.to(self.device)
            target_data_loader = torch.utils.data.DataLoader(target_dataset,batch_size=16,shuffle=False)

            for i, (imgs, labels, paths) in tqdm(enumerate(target_data_loader), total=len(target_data_loader)):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                true_labels = true_labels+labels.tolist()
                with torch.no_grad():
                    self.model.trained_model.eval()
                    out = self.model.trained_model(imgs)
                    ps = out
                    pr = [(i.tolist()).index(max(i.tolist())) for i in ps]
                    pred_labels = pred_labels+pr

            cm = metrics.confusion_matrix(true_labels, pred_labels)
            show_confusion_matrix(cm=cm,
                                  target_names=target_classes,
                                  title='Confusion Matrix',
                                  cmap=cmap,
                                  normalize=False,
                                  figure_size=figure_size
                                  )

    def roc(self, **kw):
        show_roc([self.model], **kw)
