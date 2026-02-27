import sys

import transformers

# patch for verl to support transformers v5
if not hasattr(sys.modules["transformers"], "AutoModelForVision2Seq"):
    setattr(
        sys.modules["transformers"],
        "AutoModelForVision2Seq",
        transformers.AutoModelForImageTextToText,
    )
    sys.modules["transformers"].__all__.append("AutoModelForVision2Seq")
