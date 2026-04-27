from .constants import IGNORE_INDEX


def get_entities(tags):
    start = -1
    entity_type = None
    entities = []
    for index, tag in enumerate(tags):
        if tag == "O":
            if entity_type is not None:
                entities.append((entity_type, start, index - 1))
                start = -1
                entity_type = None
            continue

        if "-" in tag:
            prefix, current_type = tag.split("-", 1)
        else:
            prefix, current_type = "B", tag

        if prefix == "B" or entity_type != current_type:
            if entity_type is not None:
                entities.append((entity_type, start, index - 1))
            start = index
            entity_type = current_type
        elif prefix == "I" and entity_type is None:
            start = index
            entity_type = current_type

    if entity_type is not None:
        entities.append((entity_type, start, len(tags) - 1))
    return entities


def entity_f1(predictions, targets, id2tag):
    correct_num = 0
    predict_num = 0
    truth_num = 0
    assert len(predictions) == len(targets)

    for pred_row, target_row in zip(predictions, targets):
        pred_tags, true_tags = [], []
        for pred_id, target_id in zip(pred_row, target_row):
            if target_id == IGNORE_INDEX:
                continue
            pred_tags.append(id2tag[pred_id])
            true_tags.append(id2tag[target_id])

        pred_entities = set(get_entities(pred_tags))
        true_entities = set(get_entities(true_tags))
        correct_num += len(pred_entities & true_entities)
        predict_num += len(pred_entities)
        truth_num += len(true_entities)

    precision = correct_num / predict_num if predict_num else 0.0
    recall = correct_num / truth_num if truth_num else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall}
