class _AttemptFailedWithPartialData(Exception):
    """
    Wraps a hard failure (network error, rate limit, etc.) that occurred
    mid-attempt — after some tool calls had already succeeded and
    returned real data. Carries that partial data along so the caller
    doesn't discard genuinely correct tool results just because the
    final completion call afterward happened to fail.
    """
    def __init__(self, original_exception, tool_result_strings, tool_call_records):
        super().__init__(str(original_exception))
        self.tool_result_strings = tool_result_strings
        self.tool_call_records = tool_call_records