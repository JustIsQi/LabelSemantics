from .constants import IGNORE_INDEX


def get_entities(tags, strict=True):
    """Decode BIO tag sequence into ``(type, start, end)`` spans.

    When ``strict=True`` (default) an ``I-X`` tag whose previous tag is not the same
    entity type is treated as ``O`` instead of opening a new span. This avoids the
    boundary-overflow case where a noisy ``I-X`` prediction at a separator (e.g. "、")
    glues two adjacent entities together or seeds a spurious one-character entity.
    """
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

        if prefix == "B":
            if entity_type is not None:
                entities.append((entity_type, start, index - 1))
            start = index
            entity_type = current_type
        elif prefix == "I" and entity_type == current_type:
            continue
        elif prefix == "I" and entity_type is not None and entity_type != current_type:
            entities.append((entity_type, start, index - 1))
            if strict:
                start = -1
                entity_type = None
            else:
                start = index
                entity_type = current_type
        elif prefix == "I" and entity_type is None:
            if strict:
                continue
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


def _prf(correct_num, predict_num, truth_num):
    precision = correct_num / predict_num if predict_num else 0.0
    recall = correct_num / truth_num if truth_num else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall}


def entity_f1_breakdown(predictions, targets, id2tag, english_masks=None):
    """Return overall, per-type, and English-company entity metrics."""
    overall_correct = 0
    overall_predict = 0
    overall_truth = 0
    type_counts = {}
    english_c_correct = 0
    english_c_predict = 0
    english_c_truth = 0
    assert len(predictions) == len(targets)

    if english_masks is None:
        english_masks = [None] * len(predictions)

    for pred_row, target_row, english_mask in zip(predictions, targets, english_masks):
        pred_tags, true_tags, mask = [], [], []
        for index, (pred_id, target_id) in enumerate(zip(pred_row, target_row)):
            if target_id == IGNORE_INDEX:
                continue
            pred_tags.append(id2tag[pred_id])
            true_tags.append(id2tag[target_id])
            mask.append(bool(english_mask[index]) if english_mask is not None else False)

        pred_entities = set(get_entities(pred_tags))
        true_entities = set(get_entities(true_tags))
        overall_correct += len(pred_entities & true_entities)
        overall_predict += len(pred_entities)
        overall_truth += len(true_entities)

        entity_types = {entity[0] for entity in pred_entities | true_entities}
        for entity_type in entity_types:
            counts = type_counts.setdefault(entity_type, {"correct": 0, "predict": 0, "truth": 0})
            pred_typed = {entity for entity in pred_entities if entity[0] == entity_type}
            true_typed = {entity for entity in true_entities if entity[0] == entity_type}
            counts["correct"] += len(pred_typed & true_typed)
            counts["predict"] += len(pred_typed)
            counts["truth"] += len(true_typed)

        def is_english_company(entity):
            entity_type, start, end = entity
            return entity_type == "C" and any(mask[start:end + 1])

        pred_english_c = {entity for entity in pred_entities if is_english_company(entity)}
        true_english_c = {entity for entity in true_entities if is_english_company(entity)}
        english_c_correct += len(pred_english_c & true_english_c)
        english_c_predict += len(pred_english_c)
        english_c_truth += len(true_english_c)

    metrics = _prf(overall_correct, overall_predict, overall_truth)
    metrics["by_type"] = {
        entity_type: _prf(counts["correct"], counts["predict"], counts["truth"])
        for entity_type, counts in sorted(type_counts.items())
    }
    metrics["english_C"] = _prf(english_c_correct, english_c_predict, english_c_truth)
    return metrics
