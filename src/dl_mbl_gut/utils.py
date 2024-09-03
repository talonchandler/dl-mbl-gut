

def center_crop(x, y):
    """Center-crop x to match spatial dimensions given by y."""

    x_target_size = x.size()[:2] + y.size()[2:]

    offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

    slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

    return x[slices]