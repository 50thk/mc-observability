post_server_error_query_description = {
    "api_description": "Run observability RCA from a natural-language query using Loki, Tempo, and InfluxDB evidence.",
    "response": {200: {"description": "Server error analysis response"}},
}

get_server_error_records_description = {
    "api_description": "List observability RCA analysis records.",
    "response": {200: {"description": "Server error analysis records"}},
}

get_server_error_record_detail_description = {
    "api_description": "Get observability RCA analysis detail by analysis ID.",
    "response": {200: {"description": "Server error analysis record"}},
}
