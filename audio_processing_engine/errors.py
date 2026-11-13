class AudioProcessingError(RuntimeError):
    """The audio could not be decoded, changed or encoded."""


class InvalidAudioError(ValueError):
    """An upload is not decodable audio in a supported format."""
