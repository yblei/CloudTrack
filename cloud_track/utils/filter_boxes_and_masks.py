import torch


def get_best_mask(masks, boxes_filt, scores):
    """Returns the mask with the highest confidence.

    Args:
        image_pil (PIL.Image): The image as a Pillow Image.
        masks (torch.Tensor): The masks of the objects in the image.
        boxes_filt (torch.Tensor): The bounding boxes of the objects
            in the image.
        pred_phrases (list): The predicted phrases.

    Returns:
        mask (torch.Tensor): The mask with the highest confidence.

    """
    # get the mask with highest confidence
    # mask = masks[0]

    # tensor from scores
    scores = torch.tensor(scores)

    # get the index of the highest score
    index = torch.argmax(scores)

    # get the mask with the highest score
    mask = masks[index, :, :, :]

    # restore the original dimensions
    mask = mask.unsqueeze(0)

    # assert no dim changes
    assert mask.dim() == masks.dim(), "Dimension has changed."

    return mask


def get_box_minimizing_cost_function(boxes_filt, scores, old_box):
    def cost_function(box1, box2):

        # get the center of the old box
        old_center = old_box[0:2]
        old_wh = old_box[2:4]

        # get the center of the new box
        new_center = box2[0:2]
        new_wh = box2[2:4]

        # calculate the distance between the centers
        distance = torch.norm(old_center - new_center)

        # calculate the difference in width and height
        wh_diff = torch.norm(old_wh - new_wh)

        # sum the two values
        cost = distance + wh_diff

        return cost

    # get the box with the lowest cost
    box = boxes_filt[0]
    box_idx = 0
    best_box = box
    min_cost = cost_function(old_box, box)

    for i in range(1, len(boxes_filt)):
        box = boxes_filt[i]
        cost = cost_function(old_box, box)
        if cost < min_cost:
            box_idx = i
            min_cost = cost
            best_box = box

    return best_box, box_idx


def get_best_box(boxes_filt, scores):
    """Returns the box with the highest confidence.

    Args:
        boxes_filt (torch.Tensor): The bounding boxes of the objects
            in the image.
        scores (list): The scores of the bounding boxes.

    Returns:
        box (torch.Tensor): The box with the highest confidence.

    """
    # get the box with highest confidence
    # box = boxes_filt[0]

    index = get_idx_of_best_box(scores)

    # get the box with the highest score
    box = boxes_filt[index]

    # restore the original dimensions
    box = box.unsqueeze(0)

    # assert no dim changes
    assert box.dim() == boxes_filt.dim(), "Dimension has changed."

    return box, index


def get_idx_of_best_box(scores):
    # tensor from scores
    scores = torch.tensor(scores)

    # get the index of the highest score
    index = torch.argmax(scores)

    return index
