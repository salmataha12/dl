
import timm
def regnety_8gf(num_classes=5, **kwargs):
    # Remove unnecessary kwargs 
    for key in list(kwargs.keys()):
        kwargs.pop(key, None)

    # Create model
    model = timm.create_model(
        'regnety_008',
        pretrained=True,
        num_classes=num_classes
    )

    return model