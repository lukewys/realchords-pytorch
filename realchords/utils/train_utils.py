class AttrDict(dict):
    """A dictionary that allows access to its keys as attributes.

    This class is added for giving argbind args to OpenRLHF.
    """

    def __init__(self, *args, **kwargs):
        if args:
            if not isinstance(args[0], (dict, list, tuple)):
                raise ValueError(
                    "AttrDict requires a dictionary or iterable of (key, value) pairs."
                )
        super().__init__(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )

    def __getattribute__(self, key):
        if key == "__dict__":
            return dict(self)
        return super().__getattribute__(key)

    @property
    def __dict__(self):
        return dict(self)
