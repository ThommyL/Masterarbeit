"""
FasterRCNN methods
"""
import gc
import os
import pickle
from typing import Tuple, Dict

import PIL
import numpy as np
import optuna
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess
from Project.VisualFeaturesBranches.ObjectDetection import MODEL_STATE_DICTIONARY_PATH
from Project.VisualFeaturesBranches.ObjectDetection.DatasetSplits import DatasetSplits
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset


# As suggested by https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html the reference implementations
# were used for training and validation. Some of the methods in this file contain lines (similar or exact) from or call
# methods of the reference implementations at https://github.com/pytorch/vision/tree/main/references/detection. Thus, as
# required by the license, the license text from https://github.com/pytorch/vision/blob/main/LICENSE is included in the
# file license_of_reference_implementation.txt.

def collate(batch):
    """
    :param batch: A batch
    :return: tuple(zip(*batch))
    """
    return tuple(zip(*batch))


def f1(precision: float, recall: float) -> float:
    """
    :param precision: The precission score
    :param recall: the recall score
    :return: 0 if precision + recall == 0, The F1 score otherwise
    """
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def get_results_path(name: str, load: str) -> str:
    """
    :param name: The experiment name
    :param load: What to load: 'model', 'grid' or 'model_all_data'
    :return: The requested object
    """
    assert load == 'model' or load == 'config' or load == 'model_all_data'
    return os.path.join(MODEL_STATE_DICTIONARY_PATH, f'model_{name}_{load}.pkl')


def get_pretrained_model(trainable_layers: int, device, small_model):
    """
    :param trainable_layers: Number of trainable layers
    :param device: Torch device
    :param small_model: If True a smaller FasterRCNN Model is returned
    :return: The pretrained model
    """

    da: DataAccess = DataAccess.instance
    num_classes = len(da.get_unique_labels_of_bounding_boxes()) + 1
    if small_model:
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            pretrained=True, trainable_backbone_layers=trainable_layers)
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True, trainable_backbone_layers=trainable_layers)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    return model

def get_winning_config_without_augmentations():
    """
    :return: The configuration that worked best in the hyperparametersearch that did not consider data augmentations.
    """
    return {
        'lr': 7.504983569215181e-05,
        'weight_decay': 0.004641719742573437,
        'trainable_layers': 5,
        'size': 1200,
        'p_horizontal_flip': 0,
        'p_vertical_flip': 0,
        'brightness_jitter': 0,
        'contrast_jitter': 0,
        'saturation_jitter': 0,
        'hue_jitter': 0,
        'p_grayscale': 0,
        'batch_size': 1,
        'num_workers': 0,
        'device': torch.device('cuda'),
        'average_span': 5,
        'nr_epochs': 10,
        'report': False,
        'curriculum': False,
        'small_model': True,
    }, 5

def get_winning_config_with_augmentations():
    """
    :return: The configuration that worked best in the hyperparametersearch that did consider data augmentations.
    """
    return {
        'lr': 2.8047202794607652e-05,
        'weight_decay': 2.676499140174696e-05,
        'trainable_layers': 5,
        'size': 1200,
        'p_horizontal_flip': 0.4169410607998048,
        'p_vertical_flip': 0.40929565177831834,
        'brightness_jitter': 0.10252158681150622,
        'contrast_jitter': 0.009561488713321058,
        'saturation_jitter': 0.004503632067624431,
        'hue_jitter': 0.002757041476332088,
        'p_grayscale': 0.12603251283193834,
        'batch_size': 1,
        'num_workers': 1,
        'device': torch.device('cuda'),
        'average_span': 5,
        'nr_epochs': 10,
        'report': False,
        'curriculum': True,
        'small_model': True
    }, 5

def train_model_on_all_data(name: str, config: Dict, train_for_epochs):
    """
    Note: Saves the result as file
    :param name: The name of the model
    :param config: The config of the model to train
    :param train_for_epochs: The number of epochs that the model should be trained for
    :return: None
    """
    if not os.path.exists(get_results_path(name, 'model_all_data')):
        # noinspection DuplicatedCode
        lr = config['lr']
        weight_decay = config['weight_decay']
        trainable_layers = config['trainable_layers']
        size = config['size']
        p_horizontal_flip = config['p_horizontal_flip']
        p_vertical_flip = config['p_vertical_flip']
        brightness_jitter = config['brightness_jitter']
        contrast_jitter = config['contrast_jitter']
        saturation_jitter = config['saturation_jitter']
        hue_jitter = config['hue_jitter']
        p_grayscale = config['p_grayscale']
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        device = config['device']
        average_span = config['average_span']
        nr_epochs = config['nr_epochs']
        report = config['report']
        small_model = config['small_model']
        curriculum = config['curriculum']

        splits: DatasetSplits = DatasetSplits(size=size, p_vertical_flip=p_vertical_flip,
                                              p_horizontal_flip=p_horizontal_flip, brightness_jitter=brightness_jitter,
                                              contrast_jitter=contrast_jitter, saturation_jitter=saturation_jitter,
                                              hue_jitter=hue_jitter, p_grayscale=p_grayscale, curriculum=curriculum)
        torch.manual_seed(0)

        model = get_pretrained_model(trainable_layers, device, small_model)
        model.train()

        optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=lr,
                                     weight_decay=weight_decay)

        for epoch in range(train_for_epochs):
            training_loader = DataLoader(splits.get_training_all(multiplier_per_epoch=1 / nr_epochs, epoch=epoch),
                                         batch_size=batch_size, collate_fn=collate, shuffle=True,
                                         num_workers=num_workers, drop_last=True)
            model = train_one_epoch(model, optimizer, training_loader, epoch, device, average_span, report)

        torch.save(model.state_dict(), get_results_path(name, 'model_all_data'))

        with open(get_results_path(name, 'config'), 'wb+') as f:
            config['trained_for_epochs'] = train_for_epochs
            pickle.dump(config, f)
        del model
        gc.collect()
    else:
        raise Exception('Already exists')


def load_model(name: str):
    """
    :param name: The name of the model
    :return: The model as requested
    """
    with open(get_results_path(name, 'config'), 'rb') as f:
        config = pickle.load(f)
    model = get_pretrained_model(config['trainable_layers'], config['device'], config['small_model'])
    model.load_state_dict(torch.load(get_results_path(name, 'model_all_data')))

    return model


def get_test_f1(evaluator):
    """
    :param evaluator: Evaluator from which to extract the precision
    :return: F1 score of the evaluator
    """
    evaluator.summarize()
    return f1(evaluator.coco_eval['bbox'].stats[0], evaluator.coco_eval['bbox'].stats[8])


def reduce_box_count(output, threshold_nms, threshold_score, min_area):
    """
    :param output: The prediction of a model
    :param threshold_nms: Non-maximum Suppression
    :param threshold_score: Minimum required score
    :param min_area: minimum required area
    :return: Output filtered by the given criteria
    """
    over_threshold_indexes = torchvision.ops.nms(output['boxes'], output['scores'], threshold_nms)
    for k in 'boxes', 'scores', 'labels':
        output[k] = output[k][over_threshold_indexes]

    new_output = dict()
    new_output['boxes'] = []
    new_output['scores'] = []
    new_output['labels'] = []
    for i in range(len(output['scores'])):
        score = output['scores'][i].tolist()
        box = output['boxes'][i].tolist()
        label = output['labels'][i].tolist()
        if score > threshold_score and (box[2] - box[0]) * (box[3] - box[1]) > min_area:
            new_output['boxes'].append(box)
            new_output['scores'].append(score)
            new_output['labels'].append(label)
    new_output['boxes'] = torch.tensor(new_output['boxes']).to(output['boxes'].device)
    new_output['scores'] = torch.tensor(new_output['scores']).to(output['scores'].device)
    new_output['labels'] = torch.tensor(new_output['labels']).to(output['labels'].device)
    return new_output


def __get_boxes(model, model_input):
    predictions = model(model_input)
    result = []
    for p in predictions:
        current = []
        for box, label in zip(p['boxes'], p['labels']):
            current.append((box.to('cpu'), label.to('cpu')))
        result.append(current)
    return result


def __tensor_image_extend_range_and_convert_to_np_array(t):
    return np.array(t.squeeze(0).to('cpu') * 255, dtype=np.uint8)


def get_artwork_with_predicted_bounding_boxes(x, bounding_boxes, labels) -> PIL.Image:
    """
    :param x: Input image that was given to the model
    :param bounding_boxes: Bounding boxes the model predicted
    :param labels: Labels the model predicted
    :return: PIL Image showing the predictions of the model
    """
    da: DataAccess = DataAccess.instance
    result = draw_bounding_boxes(torch.tensor(__tensor_image_extend_range_and_convert_to_np_array(x)),
                                 bounding_boxes, labels=[da.get_class_label_for_index(label) for label in labels],
                                 width=3)
    return PIL.Image.fromarray(np.array(result.permute(1, 2, 0), dtype=np.uint8))


def train_model(config, trial) -> Tuple[float, int]:
    """
    :param config: Training Configuration
    :param trial: The optuna trial for which this method is called
    :return: Best F1 score and epoch it was achieved in (epoch starts at 0)
    """
    # noinspection DuplicatedCode
    lr = config['lr']
    weight_decay = config['weight_decay']
    trainable_layers = config['trainable_layers']
    size = config['size']
    p_horizontal_flip = config['p_horizontal_flip']
    p_vertical_flip = config['p_vertical_flip']
    brightness_jitter = config['brightness_jitter']
    contrast_jitter = config['contrast_jitter']
    saturation_jitter = config['saturation_jitter']
    hue_jitter = config['hue_jitter']
    p_grayscale = config['p_grayscale']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    device = config['device']
    average_span = config['average_span']
    nr_epochs = config['nr_epochs']
    report = config['report']
    small_model = config['small_model']
    curriculum = config['curriculum']

    splits: DatasetSplits = DatasetSplits(size=size, p_vertical_flip=p_vertical_flip,
                                          p_horizontal_flip=p_horizontal_flip, brightness_jitter=brightness_jitter,
                                          contrast_jitter=contrast_jitter, saturation_jitter=saturation_jitter,
                                          hue_jitter=hue_jitter, p_grayscale=p_grayscale, curriculum=curriculum)

    models = []
    optimizers = []

    def refresh_loaders(e):
        training_l = []
        test_l = []
        for split_nr in range(splits.number_of_splits):
            training_l.append(DataLoader(splits.get_train(split_nr, e, nr_epochs), batch_size=batch_size,
                                         collate_fn=collate, shuffle=True, num_workers=num_workers))
            test_l.append(DataLoader(splits.get_test(split_nr), collate_fn=collate,
                                     num_workers=num_workers))
        return training_l, test_l

    for nr in range(splits.number_of_splits):
        model = get_pretrained_model(trainable_layers, device, small_model)
        model.train()
        model.to(torch.device('cpu'))
        models.append(model)
        optimizers.append(torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=lr,
                                           weight_decay=weight_decay))

    all_results = []
    for epoch in range(nr_epochs):
        training_loaders, test_loaders = refresh_loaders(epoch)
        epoch_results = []
        for nr in range(splits.number_of_splits):
            current_model = models[nr]
            current_model = train_one_epoch(current_model, optimizers[nr], training_loaders[nr], epoch, device,
                                            average_span, report)
            evaluator = __evaluate(current_model, test_loaders[nr], device, report)
            evaluator.accumulate()
            if report:
                evaluator.summarize()

            current_result = get_test_f1(evaluator)
            epoch_results.append(current_result)

            current_model.to(torch.device('cpu'))
            torch.cuda.empty_cache()
            gc.collect()
        epoch_result = sum(epoch_results) / len(epoch_results)
        all_results.append(epoch_result)

        if trial is not None:
            trial.report(epoch_result, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    result = -np.inf
    epoch = None

    for ind, r in enumerate(all_results):
        if r > result:
            result = r
            epoch = ind

    return result, epoch


def train_one_epoch(model, optimizer, data_loader, epoch, device, average_span, report):
    """
    :param model: The model which to train for one epoch
    :param optimizer: The optimizer
    :param data_loader: The Dataloader
    :param epoch: The number of the epoch
    :param device: The torch device
    :param average_span: Average span used for reporting
    :param report: Whether to show loading bar with current scores or not
    :return: model that was trained for one (further) epoch
    """
    model.train()
    model.to(device)

    progress_bar = None

    if report:
        progress_bar = tqdm(total=len(data_loader), desc=f'Preparing for training...')

    current_average = []
    all_losses = []
    avg_loss = np.inf
    previous_avg_loss = np.inf

    for index, data_tuple in enumerate(data_loader):
        images, targets = data_tuple

        del data_tuple
        gc.collect()

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = sum(model(images, targets).values())

        del images
        del targets
        gc.collect()

        current_loss = losses.to(torch.device('cpu'))
        current_average.append(current_loss)
        all_losses.append(current_loss)

        optimizer.zero_grad()

        losses.backward()

        if index % average_span == 0 and index > 0:
            previous_avg_loss = avg_loss
            avg_loss = sum(current_average) / len(current_average)
            current_average = []
        if index % 50 == 0:
            torch.cuda.empty_cache()
        optimizer.step()
        if report:
            progress_bar.set_description(
                f'epoch {epoch}: loss: {losses:3.3f} ({previous_avg_loss:3.3f}->{avg_loss:3.3f} '
                f'[{"+" if avg_loss > previous_avg_loss else "-"}{abs(avg_loss - previous_avg_loss):3.3f}])')
            progress_bar.update()

        del losses
        del index
        gc.collect()
    if report:
        progress_bar.set_description(
            f'epoch {epoch}: average loss = {sum(all_losses) / len(all_losses)}')
        progress_bar.update()
        progress_bar.close()
    return model


def __evaluate(model, data_loader, device, report):
    model.eval()

    evaluator = CocoEvaluator(get_coco_api_from_dataset(data_loader.dataset), ["bbox"])

    progress_bar = None
    if report:
        progress_bar = tqdm(total=len(data_loader), desc='Testing...')

    for images, targets in data_loader:
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in
                   model(list(image.to(device) for image in images))]
        evaluator.update({target["image_id"].item(): output for target, output in zip(targets, outputs)})
        if report:
            progress_bar.update()
        del targets
        del outputs
        gc.collect()
    if report:
        progress_bar.close()
    return evaluator


def get_config_score(config, trial):
    return train_model(config, trial)[0]
