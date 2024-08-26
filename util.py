

def subclasses(cls, just_leaf=False):
    sc = cls.__subclasses__()
    ssc = [g for s in sc for g in subclasses(s, just_leaf)]

    return [s for s in sc if not just_leaf or not s.__subclasses__()] + ssc

def subclass_where(cls, **kwargs):
    for s in subclasses(cls):
        for k, v in kwargs.items():
            if not hasattr(s, k) or getattr(s, k) != v:
                break
        else:
            return s

    attrs = ', '.join(f'{k} = {v}' for k, v in kwargs.items())
    raise KeyError(f'No subclasses of {cls.__name__} with attrs == ({attrs})')