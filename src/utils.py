
def is_error(quotation: dict):
    assert quotation is not None
    if 'error_code' not in quotation:
        return False
    return len(quotation.get('error_code')) == 0

def fix_percentage(quotation: dict, field_name: str):
    assert quotation is not None
    assert len(field_name) > 0
    if not field_name in quotation:
        return 0.0
    # handle % at end and ',' as the thousands separator
    field_value = quotation.get(field_name)
    if isinstance(field_value, str):
        val = field_value.replace(',', '').rstrip('%')
        pc = float(val) if not is_error(quotation) else 0.0
        del quotation[field_name]
        assert field_name not in quotation
        quotation[field_name] = pc
        return pc
    else:
        quotation[field_name] = field_value # assume already converted ie. float
        return field_value