import torch.hub
import time
import csv
from NounClasses55 import noun_classes
from VerbClasses55 import verb_classes
from PIL import Image
import numpy as np
import os
import os.path
from torchvision.transforms import Compose, transforms
from transforms import GroupScale, GroupCenterCrop, GroupOverSample, Stack, ToTorchFormatTensor, GroupNormalize

def main():
    repo = 'epic-kitchens/action-models'
    class_counts = (125, 352)
    segment_count = 8
    base_model = 'resnet50'

    tsn = torch.hub.load(repo, 'TSN', class_counts, segment_count, 'RGB',
                        base_model=base_model, 
                        pretrained='epic-kitchens', force_reload=True)
    '''
    trn = torch.hub.load(repo, 'TRN', class_counts, segment_count, 'RGB',
                        base_model=base_model, 
                        pretrained='epic-kitchens')
    mtrn = torch.hub.load(repo, 'MTRN', class_counts, segment_count, 'RGB',
                        base_model=base_model, 
                        pretrained='epic-kitchens')
    tsm = torch.hub.load(repo, 'TSM', class_counts, segment_count, 'RGB',
                        base_model=base_model, 
                        pretrained='epic-kitchens')

    # Show all entrypoints and their help strings
    #for entrypoint in torch.hub.list(repo):
        #print(entrypoint)
        #print(torch.hub.help(repo, entrypoint))

    batch_size = 1
    segment_count = 8
    snippet_length = 1  # Number of frames composing the snippet, 1 for RGB, 5 for optical flow
    snippet_channels = 3  # Number of channels in a frame, 3 for RGB, 2 for optical flow
    height, width = 224, 224

    inputs = torch.randn(
        [batch_size, segment_count, snippet_length, snippet_channels, height, width]
    )
    print(inputs.shape)

    # The segment and snippet length and channel dimensions are collapsed into the channel
    # dimension
    # Input shape: N x TC x H x W
    inputs = inputs.reshape((batch_size, -1, height, width))
    print(inputs.shape)
    '''

    # Transform
    net = tsn
    crop_count = 1
    backbone_arch = base_model

    if crop_count == 1:
        cropping = Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(net.input_size),
        ])
    elif crop_count == 10:
        cropping = GroupOverSample(net.input_size, net.scale_size)

    transform = Compose([
        cropping,
        Stack(roll=backbone_arch == 'BNInception'),
        ToTorchFormatTensor(div=backbone_arch != 'BNInception'),
        GroupNormalize(net.input_mean, net.input_std),
    ])

    net.eval()

    # Get every 30th frame of 60 fps video (0.5 seconds between each frame)

    dir = '../Data/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/train/P01/P01_08/'
    num_files = len([name for name in os.listdir(dir)])

    image_tmpl = 'frame_{:010d}.jpg'
    frame_nums = [x for x in range(1, num_files+1, 30)]
    
    # Analyze a segment of 8 frames (8 frames = 4 seconds video time)

    start_idx = secs = 0
    stop_idx = 8
    idxs = frame_nums[start_idx:stop_idx]
    res_file = open('ACTIONS_P01_08.txt', 'w')

    while (len(idxs) == 8):
        images = [ Image.open(os.path.join(dir, image_tmpl.format(idx))).convert('RGB') for idx in idxs ]
        process_data = transform(images)

        # Predict
        verb_logits, noun_logits = net(process_data)

        # Verb prediction
        v_preds = torch.argmax(verb_logits, dim=1)
        v_class = v_preds.item()
        #v_class, _ = torch.mode(v_preds, 0).item() # When crop_count = 10
        verb = verb_classes[str(v_class)]

        # Noun prediction
        n_preds = torch.argmax(noun_logits, dim=1)
        n_class = n_preds.item()
        #n_class, _ = torch.mode(n_preds, 0).item() # When crop_count = 10
        noun = noun_classes[str(n_class)]

        # Print results
        time_start = time.strftime('%H:%M:%S', time.gmtime(secs))
        time_end = time.strftime('%H:%M:%S', time.gmtime(secs+4))
        res = f'[{time_start} – {time_end}] {verb}, {noun}\n'
        res_file.write(res)

        secs += 4
        start_idx += 8
        stop_idx += 8
        idxs = frame_nums[start_idx:stop_idx]

    res_file.close()

    '''
    images = []
    for idx in idxs:
        image = Image.open(os.path.join(dir, image_tmpl.format(idx))).convert('RGB')
        images.append(image)

    # Transform
    net = tsn
    backbone_arch = base_model
    cropping = GroupOverSample(net.input_size, net.scale_size)
    transform = Compose([
        cropping,
        Stack(roll=backbone_arch == 'BNInception'),
        ToTorchFormatTensor(div=backbone_arch != 'BNInception'),
        GroupNormalize(net.input_mean, net.input_std),
    ])
    process_data = transform(images)
    
    # Predict
    net.eval()
    verb_logits, noun_logits = net(process_data)

    # Verb prediction
    v_preds = torch.argmax(verb_logits, dim=1)
    v_class, _ = torch.mode(v_preds, 0)
    verb = verb_classes[str(v_class.item())]
    print(verb)

    # Noun prediction
    n_preds = torch.argmax(noun_logits, dim=1)
    n_class, _ = torch.mode(n_preds, 0)
    noun = noun_classes[str(n_class.item())]
    print(noun)
    '''

    '''
    for model in [tsn]:
        model.eval()

        # You can get features out of the models
        # features = model.features(inputs)
        # and then classify those features
        # verb_logits, noun_logits = model.logits(features)
        
        # or just call the object to classify inputs in a single forward pass
        verb_logits, noun_logits = model(inputs)
        print(verb_logits.shape)

        verb_max, verb_argmax = verb_logits.data.squeeze().max(0)
        verb_class_id = verb_argmax.item()
        print(verb_classes[str(verb_class_id)])

        noun_max, noun_argmax = noun_logits.data.squeeze().max(0)
        noun_class_id = noun_argmax.item()
        print(noun_classes[str(noun_class_id)])
    '''

if __name__ == "__main__":
    main()