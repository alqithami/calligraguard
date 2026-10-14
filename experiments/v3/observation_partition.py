"""Explicit observation-level routing, independent of model predictions."""
def partition_observability(records):
    primary, observation_only = [], []
    for row in records:
        label = row['label']
        area = row['mask_pixels256']
        if label not in (0, 1) or not isinstance(area, int) or area < 0:
            raise ValueError('Invalid label or mask area')
        expected = int(bool(label and area))
        if row.get('observation_label256') != expected:
            raise ValueError('Observation label disagrees with declared mask')
        if label == 0 and area != 0:
            raise ValueError('Nonempty mask on a clean control')
        (observation_only if label and area == 0 else primary).append(row)
    return primary, observation_only
