from decouple import config

SQUAD_SELECTION_METHOD = config(
    "SQUAD_SELECTION_METHOD", default="preselect_cheapest_players"
)
supported_squad_selection_methods = ("preselect_cheapest_players", "naive")
if SQUAD_SELECTION_METHOD not in supported_squad_selection_methods:
    raise ValueError(
        f"Invalid squad selection method, must be one of {supported_squad_selection_methods}"
    )

N_WORST_TEAMS = config(
    "N_WORST_TEAMS", default=5, cast=int
)  # Number of worst teams to exclude from preselection. Only used with preselect_cheapest_players method
