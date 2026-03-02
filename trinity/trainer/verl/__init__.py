import sys

import transformers


# start of patch for verl to support transformers v5
def patch_for_transformers_v5():
    if not hasattr(sys.modules["transformers"], "AutoModelForVision2Seq"):
        setattr(
            sys.modules["transformers"],
            "AutoModelForVision2Seq",
            transformers.AutoModelForImageTextToText,
        )
        sys.modules["transformers"].__all__.append("AutoModelForVision2Seq")


patch_for_transformers_v5()
# end of patch for verl to support transformers v5
