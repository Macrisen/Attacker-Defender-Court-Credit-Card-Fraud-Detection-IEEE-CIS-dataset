"""Predeclared conservative rules; no protected evaluation input."""
def select(stats, duplicate_of=None):
    if stats['unique_count'] == 0:
        return 'DROP', 'All NULL in fitting prefix; no observed variation'
    if stats['unique_count'] == 1 and stats['missing_count'] == 0:
        return 'DROP', 'Complete constant in fitting prefix'
    if duplicate_of:
        return 'DROP', 'Exact fitting-prefix duplicate, including NULL mask, of '+duplicate_of
    if stats['unique_count'] == 1:
        return 'REVIEW', 'Constant non-NULL value but missingness can carry information'
    if stats['missing_percentage'] > 90:
        return 'REVIEW', 'High missingness; retain missingness information'
    if stats['dominant_value_percentage'] > 99.5:
        return 'REVIEW', 'Rare values may carry fraud information'
    if stats.get('max_correlation', 0) >= .98:
        return 'REVIEW', 'High correlation is not sufficient evidence to remove'
    return 'KEEP', 'No predetermined removal rule triggered'
