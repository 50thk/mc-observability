def filter_tools_by_allowlist(tools, allowed_names):
    allowed = set(allowed_names or ())
    return [
        tool
        for tool in tools or []
        if getattr(tool, "name", "") in allowed
    ]
