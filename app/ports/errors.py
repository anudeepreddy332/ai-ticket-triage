class ClassificationError(RuntimeError):
    """Base error for ticket classification failures."""


class ClassifierUnavailableError(ClassificationError):
    """The classifier dependency could not successfully complete the request."""


class ClassifierResponseError(ClassificationError):
    """The classifier returned an unusable or invalid response."""

class ClassificationBlockedError(ClassificationError):
    """Classification was blocked by a safety or policy control."""