def label(settings):
    lbl = str({k.replace('condensation_', ''):
               "{:.1e}".format(v) if type(v) is float else
               str(v).zfill(2) if type(v) is int else
               v for k,v in settings.items()})
    return lbl\
        .replace('{', '')\
        .replace('}', '')\
        .replace("'", '')\
        .replace('True', 'T')\
        .replace('False', 'F')\
        .replace('_thd', '$_{th}$')\
        .replace('e-0', 'e-')
