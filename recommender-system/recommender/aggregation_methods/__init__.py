form .borderline_based import least_misery, majority_voting, most_pleasure, most_respected_person
from .consensus_based import average, additive_utilitarian, multiplicative, average_without_misery, fairness
from .majority_based import approval_voting, plurality_voting, copeland_rule, borda_count

__all__ = [
        "least_misery", "majority_voting", "most_pleasure", "most_respected_person",
        "average", "additive_utilitarian", "multiplicative", "average_without_misery", "fairness",
        "approval_voting", "plurality_voting", "copeland_rule", "borda_count"
    ]