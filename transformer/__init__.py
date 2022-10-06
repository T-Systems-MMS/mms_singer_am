if __package__ is None or __package__ == '':
    import transformer.Constants
    import transformer.Modules
    import transformer.Layers
    import transformer.SubLayers
    import transformer.Models
else:
    from . import Constants
    from . import Modules
    from . import Layers
    from . import SubLayers
    from . import Models
