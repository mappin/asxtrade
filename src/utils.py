import json
import os

def is_error(quotation: dict):
    assert quotation is not None
    if 'error_code' not in quotation:
        return False
    return len(quotation.get('error_code')) == 0

def is_suspended(quotation: dict):
    assert quotation is not None
    if 'suspended' not in quotation:
        return False 
    return quotation.get('suspended', False) != False
        
def fix_percentage(quotation: dict, field_name: str):
    assert quotation is not None
    assert len(field_name) > 0
    if not field_name in quotation:
        return 0.0
    # handle % at end and ',' as the thousands separator
    field_value = quotation.get(field_name)
    if isinstance(field_value, str):
        val = field_value.replace(',', '').rstrip('%')
        pc = float(val) if not any([is_error(quotation), is_suspended(quotation)]) else 0.0
        del quotation[field_name]
        assert field_name not in quotation
        quotation[field_name] = pc
        return pc
    else:
        quotation[field_name] = field_value # assume already converted ie. float
        return field_value

def read_config(filename, verbose=True):
    """
    Read config.json (as specified by command line args) and return the password and mongo host
    configuration as a tuple
    """
    assert isinstance(filename, str)

    config = {}
    with open(filename, 'r') as fp:
        config = json.loads(fp.read())
        m = config.get('mongo')
        if verbose:
            print(m)
        password = m.get('password')
        if password.startswith('$'):
            password = os.getenv(password[1:])
        return m, password
